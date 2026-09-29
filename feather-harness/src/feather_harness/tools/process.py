"""Model-facing Bash and managed-task tools."""

import re
from typing import Protocol

from pydantic import Field, JsonValue, ValidationError

from ..core.messages import FrozenModel
from . import Tool, ToolResult, ToolSpec, invalid_arguments_result

_TRAILING_BARE_AMPERSAND = re.compile(r"(?<![&\\])&(?:\s*(?:#.*)?)$")

BASH_DESCRIPTION = (
    "Run a Bash command in the sandbox. Foreground commands run to completion or their deadline. "
    "With run_in_background=true, the call returns a managed task immediately; task_list, "
    "task_output, and task_stop list, inspect, and stop managed tasks. Each call starts a fresh "
    "shell process, and shell state does not carry between calls. Background mode is reserved for "
    "long-lived servers, watchers, and daemons; ordinary commands run in the foreground."
)


TASK_LIST_DESCRIPTION = (
    "List background Bash tasks started during the current session. By default, return active "
    "tasks, including failed tasks whose provider process still needs cleanup; set "
    "active_only=false to include recent completed tasks."
)


TASK_OUTPUT_DESCRIPTION = (
    "Inspect a background Bash task. By default, wait up to the supplied timeout for the task to "
    "finish; set block=false to return its current state immediately. This wait does not change "
    "the task's process deadline."
)


TASK_STOP_DESCRIPTION = (
    "Stop one background Bash task started during the current session. If the task is already "
    "finished, return its existing terminal state."
)


class BashArgs(FrozenModel):
    command: str = Field(min_length=1, description="The Bash command to execute.")
    description: str | None = Field(
        default=None,
        max_length=160,
        description=(
            "Optional concise description of the command. Used as its label when it runs in "
            "the background."
        ),
    )
    workdir: str | None = Field(
        default=None,
        description=(
            "Working directory for the command inside the workspace. Relative paths "
            "resolve from the workspace root; absolute in-workspace or explicitly "
            "granted paths are also accepted."
        ),
    )
    timeout: int | None = Field(
        default=None,
        ge=1,
        le=600_000,
        description=(
            "Process deadline in milliseconds. Foreground commands default to 120000; "
            "background commands have no deadline when omitted."
        ),
    )
    run_in_background: bool = Field(
        default=False,
        description="Whether to return immediately with a managed background task.",
    )


class TaskOutputArgs(FrozenModel):
    task_id: str = Field(min_length=1, description="Feather task identifier returned by bash.")
    block: bool = Field(
        default=True,
        description="Whether to wait for a terminal state before returning.",
    )
    timeout: int = Field(
        default=30_000,
        ge=0,
        le=600_000,
        description="Maximum time in milliseconds to wait for this inspection call.",
    )


class TaskStopArgs(FrozenModel):
    task_id: str = Field(min_length=1, description="Feather task identifier returned by bash.")


class TaskListArgs(FrozenModel):
    active_only: bool = Field(
        default=True,
        description=(
            "List only background tasks that are running or whose provider process still needs "
            "cleanup."
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of background tasks to return.",
    )


class ProcessTransport(Protocol):
    async def run(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult: ...


class ProcessTools:
    def __init__(self, transport: ProcessTransport):
        self.transport = transport

    def bash_tool(self) -> Tool:
        return self._tool("bash", BASH_DESCRIPTION, BashArgs)

    def task_list_tool(self) -> Tool:
        return self._tool("task_list", TASK_LIST_DESCRIPTION, TaskListArgs)

    def task_output_tool(self) -> Tool:
        return self._tool("task_output", TASK_OUTPUT_DESCRIPTION, TaskOutputArgs)

    def task_stop_tool(self) -> Tool:
        return self._tool("task_stop", TASK_STOP_DESCRIPTION, TaskStopArgs)

    def _tool(self, name, description, args_model) -> Tool:
        async def invoke(raw_arguments):
            try:
                args = args_model.model_validate_json(raw_arguments)
            except ValidationError as exc:
                return invalid_arguments_result(exc)
            if isinstance(args, BashArgs):
                for field, value in (("command", args.command), ("workdir", args.workdir)):
                    if value is not None and "\0" in value:
                        return ToolResult(
                            content=f"The {field} must not contain NUL bytes.",
                            error="tool_error",
                        )
                match = _TRAILING_BARE_AMPERSAND.search(args.command)
                if match and not _is_after_shell_comment(args.command, match.start()):
                    return ToolResult(
                        content="Do not end a command with '&'; set run_in_background=true instead.",
                        error="tool_error",
                    )
            return await self.transport.run(name, args.model_dump(mode="json"))

        return Tool(
            spec=ToolSpec(
                name=name, description=description, parameters=args_model.model_json_schema()
            ),
            invoke=invoke,
        )


def _is_after_shell_comment(command: str, position: int) -> bool:
    quote: str | None = None
    escaped = False
    in_comment = False
    for index, character in enumerate(command[:position]):
        if in_comment:
            if character == "\n":
                in_comment = False
            continue
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            continue
        if character == "#" and (
            index == 0 or command[index - 1].isspace() or command[index - 1] in ";&|()<>"
        ):
            in_comment = True
    return in_comment
