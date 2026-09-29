"""Install Feather in Harbor's sandbox; Harbor owns lifecycle and grading."""

import hashlib
import json
import re
import shlex
import tempfile
from importlib.metadata import version
from pathlib import Path

from feather_harness import McpServerSpec, RunResult
from feather_harness.cli import LocalConfig
from feather_harness.production import Profile
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    EnvVar,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent

from feather_eval.atif import write_atif

INSTALL_DIR = "/installed-agent/feather"
PYTHON = f"{INSTALL_DIR}/venv/bin/python"
UV_VERSION = "0.11.14"


class FeatherHarborAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = True
    ENV_VARS = [EnvVar(kwarg="api_key", env="OPENAI_API_KEY", env_fallback="OPENAI_API_KEY")]

    def __init__(
        self,
        *args,
        wheel: str,
        base_url: str | None = None,
        max_turns: int = 100,
        workspace: str = ".",
        atif_mode: str = "segments",
        profile: Profile = "prod",
        enable_image_tools: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wheel = Path(wheel).resolve()
        if atif_mode not in {"segments", "grader"}:
            raise ValueError("atif_mode must be segments or grader")
        self.atif_mode = atif_mode
        self.config = LocalConfig(
            model=self.model_name,
            base_url=base_url,
            max_turns=max_turns,
            workspace=workspace,
            mcp_servers=tuple(_harbor_mcp_spec(server) for server in self.mcp_servers),
            skills_dir=Path(self.skills_dir) if self.skills_dir else None,
            profile=profile,
            enable_image_tools=enable_image_tools,
        )

    @staticmethod
    def name() -> str:
        return "feather"

    def get_version_command(self) -> str:
        return shlex.join(
            [
                PYTHON,
                "-c",
                "from importlib.metadata import version; print(version('feather-harness'))",
            ]
        )

    async def install(self, environment: BaseEnvironment) -> None:
        if _WHEEL_FILENAME.match(self.wheel.name) is None:
            raise ValueError(
                f"Feather harness wheel filename must include a PEP 427 version: {self.wheel.name}"
            )
        with self.wheel.open("rb") as wheel:
            wheel_sha256 = hashlib.file_digest(wheel, "sha256").hexdigest()
        condition = self.config.model_dump(
            mode="json", include={"profile", "enable_image_tools"}
        )
        if condition["enable_image_tools"] is None:
            condition["enable_image_tools"] = self.config.profile == "prod"
        condition["wheel_sha256"] = wheel_sha256
        (self.logs_dir / "harness-condition.json").write_text(json.dumps(condition, indent=2))
        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "ca_certificates", "ripgrep")
        )
        await self.exec_as_root(environment, command=f"mkdir -p {INSTALL_DIR}")
        remote_wheel = f"{INSTALL_DIR}/{self.wheel.name}"
        await environment.upload_file(self.wheel, remote_wheel)
        # Python's managed installation must be outside /root so the task user can execute it.
        await self.exec_as_root(
            environment,
            command=(
                f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh && "
                f"{INSTALL_DIR}/bin/uv venv --python 3.12 {INSTALL_DIR}/venv && "
                f"{INSTALL_DIR}/bin/uv pip install --python {PYTHON} {shlex.quote(remote_wheel)}"
            ),
            env={
                "UV_INSTALL_DIR": f"{INSTALL_DIR}/bin",
                "UV_NO_MODIFY_PATH": "1",
                "UV_PYTHON_INSTALL_DIR": f"{INSTALL_DIR}/python",
            },
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        logs = str(self.environment_logs_dir)
        # No cwd override: Harbor's task image chooses the workspace via WORKDIR.
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(self.config.model_dump_json())
            prompt = Path(temporary) / "instruction.txt"
            prompt.write_text(instruction)
            await environment.upload_file(config, f"{INSTALL_DIR}/config.json")
            await environment.upload_file(prompt, f"{INSTALL_DIR}/instruction.txt")
        command = shlex.join(
            [
                f"{INSTALL_DIR}/venv/bin/feather",
                "--config",
                f"{INSTALL_DIR}/config.json",
                "-",
            ]
        )
        env = {**self.extra_env, **self.resolve_env_vars()}
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "FeatherHarborAgent requires OPENAI_API_KEY in the Harbor agent env"
            )
        await self.exec_as_agent(
            environment,
            command=(
                f"{command} < {INSTALL_DIR}/instruction.txt "
                f"> {shlex.quote(logs + '/run.json')} 2> {shlex.quote(logs + '/stderr.txt')}"
            ),
            env=env,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Harbor skips this callback if run() has already populated the context.
        self._populate_condition(context)
        output = self.logs_dir / "run.json"
        # CLI setup failures or Harbor timeouts can leave redirected output empty.
        if not output.exists() or output.stat().st_size == 0:
            return
        result = RunResult.model_validate_json(output.read_text())
        write_atif(
            result,
            self.logs_dir,
            agent=Agent(
                name=self.name(), version=version("feather-harness"), model_name=self.model_name
            ),
            session_id=self.logs_dir.parent.name,
            grader=self.atif_mode == "grader",
        )
        context.metadata = {
            **(context.metadata or {}),
            "stop_reason": result.stop_reason,
            "num_turns": result.num_turns,
            "compactions": result.compactions,
            "recoveries": result.recoveries,
            "final_answer": result.final_answer,
        }

    def _populate_condition(self, context: AgentContext) -> None:
        condition = self.logs_dir / "harness-condition.json"
        if condition.exists():
            context.metadata = {
                **(context.metadata or {}),
                "feather": json.loads(condition.read_text()),
            }


_WHEEL_FILENAME = re.compile(r"^[\w.]+\-[0-9][^-]*\-(?:[^-]+\-)?[^-]+\-[^-]+\-[^-]+\.whl$")


def _harbor_mcp_spec(server) -> McpServerSpec:
    if server.transport == "stdio":
        return McpServerSpec(
            name=server.name,
            transport="stdio",
            command=server.command,
            args=tuple(server.args),
        )
    return McpServerSpec(name=server.name, transport=server.transport, url=server.url)
