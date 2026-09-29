import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from feather_harness import (
    AssistantMessage,
    McpServerSpec,
    RunResult,
    ToolCall,
    ToolMessage,
    TraceSegment,
    UserMessage,
)
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.trial.trial import Trial

from feather_eval.harbor import FeatherHarborAgent


class LocalEnvironment:
    """Execute the agent locally; stand in for Linux root-level package installation."""

    def __init__(self, root: Path):
        self.install_dir = root / "installed-agent" / "feather"
        self.workspace = root / "workspace"
        self.workspace.mkdir()

    async def upload_file(self, source, target):
        target = Path(target.replace("/installed-agent/feather", str(self.install_dir)))
        shutil.copyfile(source, target)

    async def exec(self, command, *, env=None, cwd=None, user=None, timeout_sec=None):
        if user == "root":
            return ExecResult(
                return_code=0, stdout="apt-get" if "for manager" in command else ""
            )
        command = command.replace("/installed-agent/feather", str(self.install_dir))
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd or self.workspace,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ExecResult(
            return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )


@pytest.mark.parametrize(
    "condition",
    [
        {},
        {"profile": "prod", "enable_image_tools": True},
        {"profile": "train", "enable_image_tools": True},
    ],
)
async def test_cli_receives_literal_instruction_and_records_its_condition(tmp_path, condition):
    environment = LocalEnvironment(tmp_path)
    executable = environment.install_dir / "venv" / "bin" / "feather"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import argparse
import json
import os
import sys
from pathlib import Path
from feather_harness import AssistantMessage, RunResult, TraceSegment, UserMessage

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('prompt', choices=['-'])
args = parser.parse_args()
instruction = sys.stdin.read()
Path('received.json').write_text(json.dumps({
    'instruction': instruction,
    'config': json.loads(args.config.read_text()),
    'api_key': os.environ['OPENAI_API_KEY'],
    'path': os.environ['PATH'],
    'workspace': os.getcwd(),
}))
print(RunResult(
    stop_reason='completed', num_turns=1, compactions=0, final_answer='Final',
    segments=(TraceSegment(
        kind='execution', input=(UserMessage(content=instruction),),
        messages=(AssistantMessage(content='Final'),),
    ),),
).model_dump_json())
print('probe diagnostic', file=sys.stderr)
"""
    )
    executable.chmod(0o755)
    logs = tmp_path / "agent logs"
    logs.mkdir()
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"selected wheel")
    agent = FeatherHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="provider/model",
        wheel=str(wheel),
        base_url="http://endpoint/v1",
        workspace="/filesystem",
        extra_env={"OPENAI_API_KEY": "private-api-key"},
        mcp_servers=[
            MCPServerConfig(
                name="local", transport="stdio", command="server", args=["--token=private-mcp-key"]
            )
        ],
        **condition,
    )
    await agent.install(environment)
    installed_wheel = environment.install_dir / wheel.name
    wheel.write_bytes(b"rebuilt later")

    instruction = "Quotes ' \" and $(touch unwanted) café\n" + "x" * 140_000 + "\n\n"
    context = AgentContext()
    await agent.run(instruction, environment, context)
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)

    received = json.loads((environment.workspace / "received.json").read_text())
    assert received["instruction"] == instruction
    assert received["config"]["model"] == "provider/model"
    assert received["config"]["base_url"] == "http://endpoint/v1"
    assert received["config"]["workspace"] == "/filesystem"
    assert received["config"]["profile"] == condition.get("profile", "prod")
    assert received["config"]["enable_image_tools"] == condition.get("enable_image_tools", False)
    assert received["api_key"] == "private-api-key"
    assert received["path"] == os.environ["PATH"]
    assert Path(received["workspace"]) == environment.workspace
    assert not (environment.workspace / "unwanted").exists()
    assert (agent.logs_dir / "stderr.txt").read_text() == "probe diagnostic\n"
    recorded = json.loads((logs / "harness-condition.json").read_text())
    assert recorded == {
        "profile": received["config"]["profile"],
        "enable_image_tools": received["config"]["enable_image_tools"],
        "wheel_sha256": hashlib.sha256(installed_wheel.read_bytes()).hexdigest(),
    }
    assert context.metadata == {
        "feather": recorded,
        "stop_reason": "completed",
        "num_turns": 1,
        "compactions": 0,
        "recoveries": 0,
        "final_answer": "Final",
    }
    trajectory = json.loads((logs / "trajectory.json").read_text())
    assert trajectory["steps"][-1]["message"] == "Final"


async def test_nonzero_cli_exit_fails_the_harbor_run(tmp_path):
    environment = LocalEnvironment(tmp_path)
    executable = environment.install_dir / "venv" / "bin" / "feather"
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/bash\nprintf "provider failed\\n" >&2\nexit 7\n')
    executable.chmod(0o755)
    logs = tmp_path / "agent logs"
    logs.mkdir()
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"selected wheel")
    agent = FeatherHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="test",
        wheel=str(wheel),
        extra_env={"OPENAI_API_KEY": "test"},
    )
    await agent.install(environment)

    context = AgentContext()
    with pytest.raises(NonZeroAgentExitCodeError):
        await agent.run("task", environment, context)

    assert (agent.logs_dir / "stderr.txt").read_text() == "provider failed\n"
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)
    assert context.metadata == {
        "feather": json.loads((logs / "harness-condition.json").read_text())
    }


@pytest.mark.parametrize("mode,root_answer", [("segments", ""), ("grader", "Final")])
def test_context_reports_outcome_and_selects_the_requested_atif_view(tmp_path, mode, root_answer):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="test", wheel=str(tmp_path / "feather.whl"), atif_mode=mode
    )
    result = RunResult(
        stop_reason="completed",
        num_turns=2,
        compactions=1,
        recoveries=1,
        final_answer="Final",
        segments=(
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="task"),),
                messages=(
                    AssistantMessage(
                        tool_calls=(ToolCall(id="read", name="read", arguments="{}"),)
                    ),
                    ToolMessage(tool_call_id="read", content="file contents"),
                ),
            ),
            TraceSegment(
                kind="compaction",
                input=(UserMessage(content="Summarize"),),
                messages=(AssistantMessage(content="summary"),),
            ),
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="summary"),),
                messages=(AssistantMessage(content="Final"),),
            ),
        ),
    )
    (tmp_path / "run.json").write_text(result.model_dump_json())
    context = AgentContext()
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)
    root = json.loads((tmp_path / "trajectory.json").read_text())
    assert root["steps"][-1]["message"] == root_answer
    assert ("continued_trajectory_ref" in root) == (mode == "segments")
    assert context.metadata == {
        "stop_reason": "completed",
        "num_turns": 2,
        "compactions": 1,
        "recoveries": 1,
        "final_answer": "Final",
    }


def test_missing_result_does_not_mask_an_install_failure(tmp_path):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="test", wheel=str(tmp_path / "feather.whl")
    )
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.is_empty()


def test_adapter_maps_harbor_stdio_http_and_skills_dir(tmp_path):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path,
        model_name="test",
        wheel=str(tmp_path / "feather.whl"),
        mcp_servers=[
            MCPServerConfig(name="mail", transport="streamable-http", url="http://mail/mcp/"),
            MCPServerConfig(
                name="local",
                transport="stdio",
                command="/usr/bin/mcp-server",
                args=["--stdio"],
            ),
        ],
        skills_dir="/harbor/skills",
    )
    assert agent.config.mcp_servers == (
        McpServerSpec(name="mail", transport="streamable-http", url="http://mail/mcp/"),
        McpServerSpec(
            name="local",
            transport="stdio",
            command="/usr/bin/mcp-server",
            args=("--stdio",),
        ),
    )
    assert agent.config.skills_dir == Path("/harbor/skills")


def test_resolved_symlink_keeps_the_versioned_wheel_filename(tmp_path) -> None:
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"x")
    alias = tmp_path / "feather_harness.whl"
    alias.symlink_to(wheel)
    agent = FeatherHarborAgent(logs_dir=tmp_path, model_name="test", wheel=str(alias))
    assert agent.wheel.name == "feather_harness-0.1.0-py3-none-any.whl"


async def test_install_rejects_a_versionless_wheel_filename(tmp_path) -> None:
    wheel = tmp_path / "feather_harness.whl"
    wheel.write_bytes(b"x")
    agent = FeatherHarborAgent(logs_dir=tmp_path, model_name="test", wheel=str(wheel))
    with pytest.raises(ValueError, match="PEP 427"):
        await agent.install(LocalEnvironment(tmp_path))
