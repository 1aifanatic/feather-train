"""Parent-owned, single-level agent sessions with independent conversation state."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import Field

from .core.messages import FrozenModel, SystemMessage, UserMessage
from .core.model import ModelSettings
from .core.runner import AgentRunError, FeatherAgent, RunRequest, RunResult

logger = logging.getLogger(__name__)

PARENT_GUIDANCE = (
    "You can delegate parts of a task to subagents that work in parallel. Delegation suits "
    "substantial parts that can proceed on their own, such as reviewing separate documents, "
    "analyzing separate data sets, or researching separate questions. Small steps, work your "
    "next step depends on, and judgment that runs through the whole task are usually better "
    "done yourself.\n\n"
    "A subagent sees none of your conversation, so write each assignment as a complete brief. "
    "State the goal and why it matters, what you already know, the relevant sources, the files "
    "it may change, and what its report should contain. When finding sources or facts is part "
    "of the work, say what needs finding. Limit scope and side effects, but leave the subagent "
    "free to do the reading, calculation, and checking its assignment requires.\n\n"
    "Never let two agents write the same file at the same time. For one shared workbook, "
    "document, or deck, have subagents research or draft the parts, and give one agent, "
    "usually you, the job of assembling the file.\n\n"
    "While subagents work, continue with parts of the task they are not doing, and avoid "
    "duplicating their work. Collect results with wait_agent when you need them, and use "
    "followup_agent when a report leaves a gap. Reports reach only you, so carry what matters "
    "into your own work and final response. You remain responsible for the result, so "
    "reconcile conflicting findings and check a decisive figure or interpretation when the "
    "report does not show how it was reached. Unfinished subagents are stopped when your run "
    "ends."
)
CHILD_GUIDANCE = (
    "You are a subagent working on one assignment from a parent agent. You see nothing of the "
    "parent's conversation beyond that assignment, and the parent cannot answer questions "
    "while you work. Resolve minor ambiguity with the most reasonable reading and note it. "
    "When an uncertainty about scope, authority, or a decisive assumption would change the "
    "result, complete the parts that do not depend on it and return the question with the "
    "alternatives. Other agents may be working in the same workspace at the same time. Change "
    "only the files your assignment allows, never write a file another agent is changing, "
    "preserve changes you did not make, and build on work already present. Do not delegate "
    "further.\n\n"
    "Your final message is the report returned to the parent. Your intermediate messages and "
    "tool results are not included, and this report replaces the user handoff described "
    "earlier. Give the parent what it needs to use your result: the answer or findings, the "
    "figures and evidence behind them and how they were derived, the paths of sources you "
    "relied on and files you changed, and anything unresolved."
)
# A wait returns as soon as a child finishes, so a long default costs nothing when children
# are fast and saves parent turns when they are not.
WAIT_DEFAULT_SECONDS = 120
WAIT_MAX_SECONDS = 600
AGENT_TOOL_NAMES = frozenset(
    {"spawn_agent", "wait_agent", "followup_agent", "stop_agent", "list_agents"}
)

# The caller keeps its sandbox, model adapter, and task-affine resource ownership.
AgentSession = Callable[[FeatherAgent], Awaitable[RunResult]]
ChildRunner = Callable[[str, AgentSession], Awaitable[RunResult]]
ResultSink = Callable[[str, int, RunResult], Awaitable[None]]


class SubagentConfig(FrozenModel):
    max_concurrent: int = Field(default=3, ge=1, le=16)
    max_agents: int = Field(default=8, ge=1, le=64)
    max_turns: int = Field(default=50, ge=1)


class AgentStatus(FrozenModel):
    agent_id: str
    status: Literal["running", "completed", "failed", "cancelled", "exhausted"]
    turns: int = 0
    answer: str = ""
    error: str | None = None


@dataclass
class _Child:
    agent_id: str
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    status: AgentStatus = field(init=False)
    runs: list[RunResult] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    stopping: bool = False

    def __post_init__(self) -> None:
        self.status = AgentStatus(agent_id=self.agent_id, status="running")


class Subagents:
    """Own children for one parent run; closing waits for their resource cleanup.

    ``run_child(id, serve)`` must assemble a fresh agent and await ``serve(agent)``
    inside that child's resource scopes. Serve retains the same tools across follow-ups.
    A child is resumable only after successful completion, within this manager's lifetime.
    """

    def __init__(
        self,
        run_child: ChildRunner,
        *,
        instructions: str,
        config: SubagentConfig = SubagentConfig(),
        model_settings: ModelSettings = ModelSettings(),
        save_result: ResultSink | None = None,
    ) -> None:
        self.config = config
        self._run_child = run_child
        self._instructions = instructions + "\n\n" + CHILD_GUIDANCE
        self._settings = model_settings
        self._save_result = save_result
        self._children: dict[str, _Child] = {}
        self._open = False

    async def __aenter__(self) -> Subagents:
        if self._open or self._children:
            raise RuntimeError("Use a new Subagents instance for each parent run")
        self._open = True
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self._open = False
        closing = asyncio.create_task(self._close())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled and exc_type is None:
            raise asyncio.CancelledError

    def spawn(self, prompt: str) -> AgentStatus:
        self._check_capacity()
        if len(self._children) >= self.config.max_agents:
            raise ValueError("This parent run has reached its total subagent limit")
        child = _Child(agent_id=uuid4().hex)
        child.queue.put_nowait(_prompt(prompt))
        self._children[child.agent_id] = child
        child.task = asyncio.create_task(self._worker(child), name=f"agent-{child.agent_id}")
        return child.status

    def followup(self, agent_id: str, prompt: str) -> AgentStatus:
        self._check_capacity()
        child = self._child(agent_id)
        if (
            child.status.status != "completed"
            or child.stopping
            or child.task is None
            or child.task.done()
        ):
            raise ValueError("Only a successfully completed, open child can receive a follow-up")
        if child.status.turns >= self.config.max_turns:
            raise ValueError("This child has exhausted its lifetime turn budget")
        prompt = _prompt(prompt)
        child.done.clear()
        child.status = child.status.model_copy(update={"status": "running", "answer": ""})
        child.queue.put_nowait(prompt)
        return child.status

    def list(self) -> tuple[AgentStatus, ...]:
        return tuple(child.status for child in self._children.values())

    def results(self, agent_id: str) -> tuple[RunResult, ...]:
        return tuple(self._child(agent_id).runs)

    async def wait(
        self, agent_ids: tuple[str, ...], *, timeout: float = WAIT_DEFAULT_SECONDS
    ) -> tuple[AgentStatus, ...]:
        if not agent_ids or not 0 <= timeout <= WAIT_MAX_SECONDS:
            raise ValueError(
                f"Supply agent IDs and a wait timeout between 0 and {WAIT_MAX_SECONDS} seconds"
            )
        children = [self._child(agent_id) for agent_id in agent_ids]
        if timeout and all(not child.done.is_set() for child in children):
            waiters = [asyncio.create_task(child.done.wait()) for child in children]
            try:
                await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiter in waiters:
                    waiter.cancel()
                await asyncio.gather(*waiters, return_exceptions=True)
        return tuple(child.status for child in children)

    async def stop(self, agent_id: str) -> AgentStatus:
        child = self._child(agent_id)
        _request_stop(child)
        if child.task is not None and not child.task.done():
            await asyncio.shield(asyncio.gather(child.task, return_exceptions=True))
            if child.status.status == "running":
                child.status = child.status.model_copy(update={"status": "cancelled"})
            child.done.set()
        return child.status

    def compaction_state(self) -> tuple[str, ...]:
        if not self._children:
            return ()
        listing = "\n".join(f"- {s.agent_id}: {s.status}" for s in self.list())
        return ("## Subagents\n\n" + listing + "\nUse wait_agent to retrieve their results.",)

    async def _worker(self, child: _Child) -> None:
        try:
            await self._run_child(child.agent_id, lambda agent: self._serve(child, agent))
            if child.status.status == "running":
                raise RuntimeError("Child runner returned without completing its assignment")
        except asyncio.CancelledError:
            if child.status.status == "running":
                child.status = child.status.model_copy(update={"status": "cancelled"})
        except Exception as exc:
            child.status = child.status.model_copy(update={"status": "failed", "error": str(exc)})
            logger.warning("Subagent %s failed: %s", child.agent_id, exc)
        finally:
            child.done.set()

    async def _serve(self, child: _Child, agent: FeatherAgent) -> RunResult:
        if AGENT_TOOL_NAMES.intersection(agent.tools):
            raise ValueError("Child agents must not receive delegation tools")
        context = (SystemMessage(content=self._instructions),)
        while (prompt := await child.queue.get()) is not None:
            request = RunRequest(
                messages=(*context, UserMessage(content=prompt)),
                max_turns=self.config.max_turns - child.status.turns,
                model_settings=self._settings,
            )
            try:
                result = await agent.run(request)
            except AgentRunError as exc:
                await self._record(child, exc.result)
                raise
            await self._record(child, result)
            if result.stop_reason != "completed":
                return result
            # The flattened audit trace includes history deliberately removed by compaction.
            segment = result.segments[-1]
            context = (*segment.input, *segment.messages)
            child.done.set()
        return child.runs[-1]

    async def _record(self, child: _Child, result: RunResult) -> None:
        child.runs.append(result)
        if self._save_result is not None:
            await self._save_result(child.agent_id, len(child.runs), result)
        status = "completed" if result.stop_reason == "completed" else "failed"
        if result.stop_reason == "MaxTurnsExceeded":
            status = "exhausted"
        child.status = AgentStatus(
            agent_id=child.agent_id,
            status=status,
            turns=child.status.turns + result.num_turns,
            answer=result.final_answer,
            error=result.stop_message,
        )

    async def _close(self) -> None:
        for child in self._children.values():
            _request_stop(child)
        await asyncio.gather(
            *(child.task for child in self._children.values() if child.task is not None),
            return_exceptions=True,
        )
        for child in self._children.values():
            # A task cancelled before its first instruction cannot run its own finally.
            if child.status.status == "running":
                child.status = child.status.model_copy(update={"status": "cancelled"})
            child.done.set()

    def _child(self, agent_id: str) -> _Child:
        try:
            return self._children[agent_id]
        except KeyError:
            raise ValueError("Unknown agent ID for this parent run") from None

    def _check_capacity(self) -> None:
        if not self._open:
            raise RuntimeError("Subagents must be used inside its async context")
        if sum(s.status == "running" for s in self.list()) >= self.config.max_concurrent:
            raise ValueError("Subagent concurrency limit reached; wait for an existing child")


def _request_stop(child: _Child) -> None:
    # Stop and parent teardown can overlap; a second cancel would interrupt cleanup.
    if child.stopping:
        return
    child.stopping = True
    if child.task is None or child.task.done():
        return
    if child.status.status == "running":
        child.task.cancel()
    else:
        child.queue.put_nowait(None)


def _prompt(value: str) -> str:
    if not value.strip():
        raise ValueError("The subagent assignment must not be empty")
    return value
