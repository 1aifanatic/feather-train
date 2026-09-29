"""Independent E2B tool calls with private, host-side result storage."""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
import uuid

from e2b import AsyncSandbox
from feather_harness.tools import ToolResult
from pydantic import JsonValue

from feather_train.sandbox.collector import Collector, _drain
from feather_train.sandbox.paths import WORKSPACE_DIR
from feather_train.sandbox.retries import write_file

logger = logging.getLogger(__name__)


class E2BRuntime:
    """Borrow a provisioned sandbox; leave container removal to its owner."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        *,
        container: str = "agent",
        workspace: str = WORKSPACE_DIR,
        runtime: str = "/opt/feather",
        log_directory: str = "/app/.feather/tasks",
        read_grants: tuple[str, ...] = (),
        write_grants: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        run_as: str | None = None,
        host_runtime: str = "/opt/feather-host",
    ):
        self.sandbox = sandbox
        self.container = container
        self.workspace = posixpath.normpath(workspace)
        if not self.workspace.startswith("/"):
            raise ValueError("workspace must be absolute")
        if any(
            not path.startswith("/") or posixpath.normpath(path) == "/"
            for path in (*read_grants, *write_grants)
        ):
            raise ValueError("grants must be absolute and cannot cover the filesystem root")
        read_grants = tuple(posixpath.normpath(path) for path in read_grants)
        write_grants = tuple(posixpath.normpath(path) for path in write_grants)
        self.runtime = runtime
        self.host_runtime = host_runtime
        self.config = {
            "workspace": self.workspace,
            "log_directory": log_directory,
            "read_grants": list(read_grants),
            "write_grants": list(write_grants),
            "grants": [*read_grants, *write_grants],
        }
        self.env = env or {}
        self.run_as = run_as
        self.namespace = uuid.uuid4().hex
        self.collector = Collector(sandbox, self.namespace, host_runtime)
        self.tasks: dict[str, str] = {}
        self.active: set[str] = set()

    async def save_transcript(self, content: str) -> None:
        await self.run("_save_transcript", {"content": content})

    async def __aenter__(self):
        return self

    async def run(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        if name == "task_list":
            return await self._task_list(arguments)
        if name in {"task_output", "task_stop"}:
            token = self.tasks.get(arguments["task_id"])
            if token is None:
                return ToolResult(
                    content=f"Task {arguments['task_id']} was not found.", error="tool_error"
                )
            if name == "task_stop":
                if token in self.active:
                    await self._stop([token])
                state = await self.collector.inspect(token, wait=True, wait_ms=30_000)
                if not state["done"]:
                    raise TimeoutError("Stopped tool did not publish its final result")
            else:
                state = await self.collector.inspect(
                    token, wait=arguments["block"], wait_ms=arguments["timeout"], fresh=True
                )
            return self._result(token, state)
        if name in {"edit", "write"}:
            path = posixpath.normpath(
                posixpath.join(self.workspace, arguments["file_path"].replace("\\", "/"))
            )
            key = (
                posixpath.relpath(path, self.workspace)
                if posixpath.commonpath((path, self.workspace)) == self.workspace
                else path
            )
            arguments = {
                **arguments,
                "revisions": {key: arguments["revisions"][key]}
                if key in arguments["revisions"]
                else {},
            }
        background = name == "bash" and arguments["run_in_background"]
        if background:
            active = [token for token in self.tasks.values() if token in self.active]
            if len(active) >= 32:
                states = await self._states(active)
                if sum(not state["done"] for state in states) >= 32:
                    return ToolResult(
                        content="At most 32 background tasks may run at once.", error="tool_error"
                    )
        token = uuid.uuid4().hex
        task_id = f"task_{len(self.tasks) + 1}" if background else "foreground"
        request = {
            "id": token,
            "name": name,
            "arguments": arguments,
            "config": self.config,
            "background": background,
            "task_id": task_id,
            "log_file": f"{self.config['log_directory']}/run_{self.namespace}_{task_id if background else token}.log",
            "docker": self._docker("python3", "-u", f"{self.runtime}/runtime/call.py", token=token),
        }
        command = await self.collector.prepare_launch(token, request)
        self.active.add(token)
        timeout = 30 if background else (arguments.get("timeout") or 120_000) / 1000 + 30
        state = await self.collector.execute(
            token, background, asyncio.get_running_loop().time() + timeout, command
        )
        result = self._result(token, state)
        if background and result.error is None:
            self.tasks[task_id] = token
        return result

    async def __aexit__(self, exc_type, exc, traceback):
        closing = asyncio.create_task(self._close())
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            try:
                await _drain(closing)
            except Exception:
                logger.exception("E2B tool cleanup failed during cancellation")
            raise
        except Exception:
            if exc_type is None:
                raise
            logger.exception("E2B tool cleanup failed")

    def _docker(self, *command: str, token: str | None = None) -> list[str]:
        prefix = ["docker", "exec", "-i", "-w", self.workspace]
        if self.run_as is not None:
            prefix.extend(("-u", self.run_as))
        for name, value in self.env.items():
            prefix.extend(("-e", f"{name}={value}"))
        if token is not None:
            prefix.extend(("-e", f"FEATHER_PROCESS_TOKEN={token}-supervisor"))
        return [*prefix, self.container, *command]

    def _result(self, token: str, state) -> ToolResult:
        if "failure" in state:
            raise RuntimeError(state["failure"])
        result = ToolResult.model_validate(state["result"])
        if state["done"]:
            self.active.discard(token)
        return result

    async def _states(self, tokens):
        if not tokens:
            return []
        path = self.collector.path(uuid.uuid4().hex)
        await write_file(
            self.sandbox,
            path,
            json.dumps({"paths": [self.collector.path(token) for token in tokens]}).encode(),
        )
        response = await self.collector.read(
            shlex.join(["python3", f"{self.host_runtime}/remote.py", "tasks", path]), timeout=30
        )
        states = json.loads(response)
        for token, state in zip(tokens, states, strict=True):
            if "failure" in state:
                raise RuntimeError(state["failure"])
            if state["done"]:
                self.active.discard(token)
        return states

    async def _task_list(self, args) -> ToolResult:
        tokens = [
            token
            for token in self.tasks.values()
            if token in self.active or not args["active_only"]
        ]
        states = await self._states(tokens)
        selected = [state for state in states if not state["done"] or not args["active_only"]]
        heading = "Active background tasks" if args["active_only"] else "Background tasks"
        if not selected:
            return ToolResult(content=heading + ": none")
        total = len(selected)
        selected = selected[-args["limit"] :]
        content = (
            f"Showing {len(selected)} of {total} {heading.lower()}:"
            if len(selected) < total
            else f"{heading}: {total}"
        )
        return ToolResult(
            content=content + "\n" + "\n".join(state["list_entry"] for state in selected)
        )

    async def _stop(self, tokens: list[str]) -> None:
        if tokens:
            try:
                # A provider launch acknowledgement precedes docker exec. Wait
                # until the container supervisor exists so cleanup cannot miss it.
                await self.sandbox.commands.run(
                    shlex.join(
                        [
                            "python3",
                            f"{self.host_runtime}/remote.py",
                            "started",
                            *[self.collector.path(token) for token in tokens],
                        ]
                    ),
                    user="root",
                    timeout=15,
                )
            finally:
                await self.sandbox.commands.run(
                    shlex.join(self._docker("python3", f"{self.runtime}/runtime/call.py", *tokens)),
                    user="root",
                    timeout=120,
                )

    async def _close(self) -> None:
        try:
            await self._stop(list(self.active))
        finally:
            await self.sandbox.commands.run(
                shlex.join(
                    [
                        "python3",
                        f"{self.host_runtime}/remote.py",
                        "finish",
                        f"/tmp/feather-{self.namespace}",
                    ]
                ),
                user="root",
                timeout=30,
            )
