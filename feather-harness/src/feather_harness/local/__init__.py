"""Assemble one local Feather run from the shared loop and tools."""

import asyncio
from pathlib import Path

from ..core.compaction import CompactionConfig
from ..core.messages import SystemMessage, UserMessage
from ..core.model import ChatCompletionsModel, ModelSettings
from ..core.runner import FeatherAgent, RunRequest, RunResult
from ..production import (
    Profile,
    build_production_instructions,
    production_compaction,
    production_tools,
    validate_profile,
)
from ..prompt import build_instructions
from ..scripts.install_skills import install_production_skills, install_skills, install_task_skills
from ..scripts.save_transcript import save_transcript
from ..tools.files import FileTools
from ..tools.mcp import McpServerSpec, run_with_mcp
from ..tools.process import ProcessTools
from ..tools.todo import TodoList
from .filesystem import LocalFileRuntime
from .process import LocalShellRuntime


async def run_local(
    prompt: str,
    *,
    model: ChatCompletionsModel,
    workspace: Path,
    max_turns: int = 100,
    model_settings: ModelSettings = ModelSettings(),
    compaction: CompactionConfig = CompactionConfig(),
    mcp_servers: tuple[McpServerSpec, ...] = (),
    skills_dir: Path | None = None,
    profile: Profile = "prod",
    enable_image_tools: bool | None = None,
) -> RunResult:
    validate_profile(profile)
    if enable_image_tools is None:
        enable_image_tools = profile == "prod"
    file_transport = LocalFileRuntime(workspace)
    files = FileTools(file_transport, enable_image_reads=profile == "prod" and enable_image_tools)
    workspace = file_transport.workspace
    # Bundled skills are harness-owned and replaced at every start, so a workspace reused
    # across profiles carries this run's manuals rather than another profile's copies.
    install_skills(workspace, overwrite=True)
    if profile == "prod":
        install_production_skills(workspace)
    if skills_dir is not None:
        install_task_skills(workspace, skills_dir)
    if profile == "prod":
        instructions = build_production_instructions(
            skills_root=workspace / ".agents", enable_image_tools=enable_image_tools
        )
        compaction = production_compaction(compaction)
    else:
        instructions = build_instructions(skills_root=workspace / ".agents")
    request = RunRequest(
        messages=(
            SystemMessage(content=instructions),
            UserMessage(content=prompt),
        ),
        max_turns=max_turns,
        model_settings=model_settings,
    )
    todos = TodoList()

    async def publish_transcript(content: str) -> None:
        await asyncio.to_thread(save_transcript, workspace / ".feather/transcript.jsonl", content)

    async with LocalShellRuntime(
        workspace, log_directory=workspace / ".feather/tasks"
    ) as transport:
        processes = ProcessTools(transport)
        builtins = (
            processes.bash_tool(),
            processes.task_list_tool(),
            processes.task_output_tool(),
            processes.task_stop_tool(),
            files.read_tool(),
            files.edit_tool(),
            files.write_tool(),
            files.grep_tool(),
            files.glob_tool(),
            todos.tool(),
        )

        if enable_image_tools:
            builtins = (*builtins, files.view_image_tool())
        if profile == "prod":
            builtins = production_tools(builtins, enable_image_tools=enable_image_tools)

        async def run(mcp_tools):
            agent = FeatherAgent(
                model=model,
                tools=(*mcp_tools, *builtins),
                compaction=compaction,
                compaction_tool_state=todos.compaction_state,
                save_transcript=publish_transcript,
                response_recovery=profile == "prod",
            )
            return await agent.run(request)

        return await run_with_mcp(
            mcp_servers,
            run,
            reserved_names=frozenset(tool.spec.name for tool in builtins),
            enable_image_tools=enable_image_tools,
            file_transport=file_transport,
        )
