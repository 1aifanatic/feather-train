"""Delegation tools; ordinary tools retain their existing sequential execution."""

import json
from collections.abc import Awaitable, Callable

from pydantic import Field, ValidationError

from ..core.messages import FrozenModel
from ..subagents import WAIT_DEFAULT_SECONDS, WAIT_MAX_SECONDS, Subagents
from . import Tool, ToolResult, ToolSpec, invalid_arguments_result


class SpawnArgs(FrozenModel):
    prompt: str = Field(
        min_length=1,
        description="Complete brief with the goal, what you know, the relevant sources, the files the subagent may change, and what its report should contain. The subagent does not see your conversation.",
    )


class AgentArgs(FrozenModel):
    agent_id: str = Field(min_length=1)


class FollowupArgs(AgentArgs):
    prompt: str = Field(min_length=1)


class WaitArgs(FrozenModel):
    agent_ids: tuple[str, ...] = Field(min_length=1)
    timeout: float = Field(
        default=WAIT_DEFAULT_SECONDS,
        ge=0,
        le=WAIT_MAX_SECONDS,
        description="Seconds to wait for any selected subagent to finish. Returns as soon as one does, and zero inspects immediately. A timeout does not stop the work.",
    )


class ListArgs(FrozenModel):
    pass


def subagent_tools(agents: Subagents) -> tuple[Tool, ...]:
    async def spawn(args: SpawnArgs) -> dict[str, object]:
        return agents.spawn(args.prompt).model_dump(mode="json")

    async def followup(args: FollowupArgs) -> dict[str, object]:
        return agents.followup(args.agent_id, args.prompt).model_dump(mode="json")

    async def wait(args: WaitArgs) -> list[dict[str, object]]:
        return [
            s.model_dump(mode="json")
            for s in await agents.wait(args.agent_ids, timeout=args.timeout)
        ]

    async def stop(args: AgentArgs) -> dict[str, object]:
        return (await agents.stop(args.agent_id)).model_dump(mode="json")

    async def listing(args: ListArgs) -> list[dict[str, object]]:
        return [s.model_dump(mode="json", exclude={"answer"}) for s in agents.list()]

    return (
        _tool(
            "spawn_agent",
            "Start a subagent on a self-contained assignment and return its ID immediately. The subagent works in the shared workspace, sees only the assignment, and cannot delegate further. Collect its report with wait_agent.",
            SpawnArgs,
            spawn,
        ),
        _tool(
            "wait_agent",
            "Return the status and latest report of selected subagents, waiting up to the timeout for any of them to finish. A timeout does not stop them. Wait when you need a result, and do other work in between instead of waiting repeatedly.",
            WaitArgs,
            wait,
        ),
        _tool(
            "followup_agent",
            "Send another assignment to a successfully completed child, preserving its context and tools. Running, stopped, failed, or exhausted children cannot be resumed.",
            FollowupArgs,
            followup,
        ),
        _tool(
            "stop_agent",
            "Stop one owned child and wait for its resource cleanup. Stopping a completed child closes its session without changing its result.",
            AgentArgs,
            stop,
        ),
        _tool(
            "list_agents",
            "List this run's children and their status. Use wait_agent for answers. Children are stopped when the parent run ends.",
            ListArgs,
            listing,
        ),
    )


def _tool[T: FrozenModel](
    name: str, description: str, schema: type[T], invoke: Callable[[T], Awaitable[object]]
) -> Tool:
    async def call(raw: str) -> ToolResult:
        try:
            args = schema.model_validate_json(raw)
        except ValidationError as exc:
            return invalid_arguments_result(exc)
        try:
            result = await invoke(args)
        except ValueError as exc:
            return ToolResult(content=str(exc), error="tool_error")
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    return Tool(
        spec=ToolSpec(name=name, description=description, parameters=schema.model_json_schema()),
        invoke=call,
    )
