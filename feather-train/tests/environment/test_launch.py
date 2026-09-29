"""Exercise native launch assembly while controlling external services."""

# ruff: noqa: E402
import json
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from feather_harness import CompactionConfig, Tool, ToolSpec
from feather_harness.prompt import build_instructions
from feather_harness.tools import ToolResult
from feather_harness.tools.mcp import McpServerSpec
from openai import AsyncOpenAI

vf = pytest.importorskip("verifiers.v1")

from feather_train import harness as harness_module
from feather_train.grading.grader import GradingConfig
from feather_train.grading.rewards import TrajectoryRewardConfig
from feather_train.grading.types import GradeResult
from feather_train.harness import FeatherHarness, FeatherHarnessConfig
from feather_train.sandbox.runtime import build_runtime, install_runtime
from feather_train.sandbox.transport import E2BRuntime
from feather_train.tasks.taskset import FeatherTaskData


@pytest.mark.parametrize("submission_transport", ["submit_tool", "text_response"])
@pytest.mark.parametrize("runtime_source", ["default", "explicit"])
async def test_launch_isolates_before_grading_and_records_shaped_reward(
    sandbox, monkeypatch, submission_transport, tmp_path, runtime_source
):
    archive = tmp_path / "explicit-runtime.tar.gz"
    if runtime_source == "explicit":
        build_runtime(archive)
    built_archives = []

    def build(path, *, profile="train"):
        built_archives.append(path)
        build_runtime(path, profile=profile)

    monkeypatch.setattr(harness_module, "build_runtime", build)
    task = FeatherTaskData(
        idx=0,
        task_id="test",
        task_name="test",
        world_id="test",
        hf_dataset="unused/test",
        hf_revision="a" * 40,
        prompt=[{"role": "user", "content": "Solve exactly this task."}],
        system_prompt="Follow the task.",
        rubric=[],
        rubric_doc={},
        expected_deliverable={},
        submission_transport=submission_transport,
        grading_evidence_mode="submit_tool"
        if submission_transport == "submit_tool"
        else "direct_text",
        requires_web_search=False,
        network_mode="public",
        mcp_servers=[],
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="FeatherTask", data=task),
        agent=vf.AgentInfo(model="test", sampling=vf.Sampling()),
    )
    harness = FeatherHarness(
        FeatherHarnessConfig(
            runtime_archive=archive if runtime_source == "explicit" else None,
            enable_braintrust=False,
            tools=("todo_write", "write"),
            compaction=CompactionConfig(enabled=False),
            grading=GradingConfig(stage_agent_trace=True),
            trajectory_reward=TrajectoryRewardConfig(invalid_tool_call_penalty=0.1),
        )
    )
    servers = [
        McpServerSpec(name=name, transport="http", url=f"http://{name}.invalid")
        for name in ("calendar", "submission")
    ]
    phases = []
    requests = []
    discovered = []
    verifier = object()
    sandbox.sandbox_id = "test-sandbox"

    async def prepare_grading(task):
        notes = await sandbox.commands.run(
            f"docker exec {sandbox.container} cat /filesystem/notes.txt"
        )
        assert notes.stdout == "working notes"
        phases.append("isolation")
        return verifier

    @asynccontextmanager
    async def start(task, **kwargs):
        yield SimpleNamespace(
            sandbox=sandbox,
            runtime=SimpleNamespace(mcp_servers=servers),
            prepare_grading=prepare_grading,
        )
        phases.append("closed")

    async def discover(servers, run, **kwargs):
        discovered.extend(servers)

        async def invoke(arguments):
            return ToolResult(content="accepted")

        return await run(
            tuple(
                Tool(ToolSpec(name=server.name, description="Remote tool", parameters={}), invoke)
                for server in servers
            )
        )

    def respond(request):
        assert not phases, "The agent must finish before grading isolation"
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "The final answer."}
        if len(requests) == 1:
            message["tool_calls"] = [
                {
                    "id": "write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"file_path":"notes.txt","content":"working notes"}',
                    },
                },
                {
                    "id": "invalid",
                    "type": "function",
                    "function": {"name": "todo_write", "arguments": '{"todos":"not a list"}'},
                },
                {
                    "id": "unknown",
                    "type": "function",
                    "function": {"name": "invented", "arguments": "{}"},
                },
            ]
        return httpx.Response(
            200,
            json={
                "id": "response",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                    }
                ],
            },
        )

    async def grade(task, target, *, instruction, final_answer, agent_evidence, **kwargs):
        assert instruction == "Solve exactly this task."
        assert phases == ["isolation"]
        assert target is verifier
        assert final_answer == agent_evidence.trace.final_answer == "The final answer."
        assert agent_evidence.model == "test"
        assert agent_evidence.instructions == build_instructions("Follow the task.")
        assert agent_evidence.trace.num_turns == 2
        assert [
            message.tool_call_id
            for message in agent_evidence.trace.execution_messages()
            if message.role == "tool"
        ] == ["write", "invalid", "unknown"]
        phases.append("grading")
        return GradeResult(
            score=0.8, num_met=4, num_criteria=5, criteria=(), stdout="", failed_attempts=[]
        )

    monkeypatch.setattr(harness_module.ComposeSandbox, "start", start)
    monkeypatch.setattr(
        harness_module,
        "install_runtime",
        partial(install_runtime, container=sandbox.container, host_runtime=sandbox.host_runtime),
    )
    monkeypatch.setattr("feather_train.agent.run_with_mcp", discover)
    monkeypatch.setattr(
        harness_module,
        "E2BRuntime",
        partial(
            E2BRuntime,
            container=sandbox.container,
            host_runtime=sandbox.host_runtime,
        ),
    )
    def client(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )

    monkeypatch.setattr(harness_module, "AsyncOpenAI", client)
    monkeypatch.setattr(harness_module, "grade_task", grade)
    await sandbox.commands.run(f"docker exec -u root {sandbox.container} rm -rf /opt/feather")
    Path(sandbox.host_runtime, "remote.py").unlink()
    for attempt in range(2):
        if attempt:
            # A restored snapshot may contain an older, unusable tool implementation.
            await sandbox.commands.run(
                f"docker exec -u root {sandbox.container} truncate -s 0 "
                "/opt/feather/runtime/write_tool.py /filesystem/.agents/docx/SKILL.md"
            )
        requests.clear()
        phases.clear()
        discovered.clear()
        trace = vf.Trace(task=trace.task, agent=trace.agent)
        uploads_before = len(sandbox.files.written_paths)
        result = await harness.launch(
            SimpleNamespace(
                model="test",
                sampling=vf.Sampling(
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=131072,
                    top_k=0,
                    provider={"only": ["baseten"]},
                ),
            ),
            trace,
            None,
            "https://model.invalid/v1",
            "secret",
            {},
        )

        assert result.exit_code == 0
        assert phases == ["isolation", "grading", "closed"]
        assert requests[0]["messages"][-1]["content"] == "Solve exactly this task."
        assert {key: requests[0][key] for key in ("temperature", "top_p", "max_tokens", "top_k", "provider")} == {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 131072,
            "top_k": 0,
            "provider": {"only": ["baseten"]},
        }
        names = {tool["function"]["name"] for tool in requests[0]["tools"]}
        assert names == (
            {"todo_write", "write", "calendar", "submission"}
            if submission_transport == "submit_tool"
            else {"todo_write", "write", "calendar"}
        )
        assert [(server.name, server.raise_on_error_result) for server in discovered] == (
            [("calendar", False), ("submission", True)]
            if submission_transport == "submit_tool"
            else [("calendar", False)]
        )
        assert trace.info["tool_call_count"] == 3
        assert trace.info["tool_error_schema_invalid"] == trace.info["tool_error_unknown_tool"] == 1
        assert trace.info["grading_score"] == 0.8
        assert trace.info["training_reward"] == pytest.approx(0.6)
        assert trace.info["trajectory_tool_penalty"] == pytest.approx(0.2)
        assert len(sandbox.files.written_paths) == uploads_before + 1
        await sandbox.commands.run(
            f"docker exec {sandbox.container} cmp "
            "/opt/feather/skills/docx/SKILL.md /filesystem/.agents/docx/SKILL.md"
        )

    assert len(built_archives) == (1 if runtime_source == "default" else 0)
