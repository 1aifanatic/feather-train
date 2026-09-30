from __future__ import annotations

import asyncio

import pytest

from feather_harness import AssistantMessage, CompactionConfig, FeatherAgent, RunResult, ToolResult
from feather_harness.subagents import (
    AgentSession,
    ChildRunner,
    SubagentConfig,
    Subagents,
    render_subagent_reports,
)
from feather_harness.tools.agents import subagent_tools

from ..helpers import ScriptedModel, tool, work_message


async def test_children_overlap_and_wait_returns_when_only_one_finishes() -> None:
    both_started = asyncio.Event()
    releases: dict[str, asyncio.Event] = {}

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        async def invoke(raw: str) -> ToolResult:
            releases[agent_id] = asyncio.Event()
            if len(releases) == 2:
                both_started.set()
            await releases[agent_id].wait()
            return ToolResult(content="finished work")

        return await serve(
            FeatherAgent(
                model=ScriptedModel(work_message(), AssistantMessage(content=agent_id)),
                tools=(tool(invoke),),
            )
        )

    async with Subagents(run_child, instructions="shared instructions") as agents:
        first = agents.spawn("first assignment").agent_id
        second = agents.spawn("second assignment").agent_id
        await asyncio.wait_for(both_started.wait(), 2)
        releases[first].set()

        statuses = await agents.wait((first, second), timeout=1)

        assert [(status.agent_id, status.status) for status in statuses] == [
            (first, "completed"),
            (second, "running"),
        ]
        repeated = asyncio.create_task(agents.wait((first, second), timeout=1))
        try:
            await asyncio.sleep(0.01)
            assert not repeated.done()
            assert (await agents.wait((first, second), timeout=0))[1].status == "running"
            releases[second].set()
            assert (await asyncio.wait_for(repeated, 1))[1].answer == second
            assert all(
                s.status == "completed"
                for s in await asyncio.wait_for(agents.wait((first, second)), 1)
            )
        finally:
            repeated.cancel()
            await asyncio.gather(repeated, return_exceptions=True)


async def test_finished_followup_is_new_even_with_another_child_still_running() -> None:
    followed_up = asyncio.Event()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        if agent_id == blocked:
            await asyncio.Future()
        model = ScriptedModel(AssistantMessage(content="first"), AssistantMessage(content="second"))
        return await serve(FeatherAgent(model=model))

    async def save_result(agent_id: str, run_number: int, result: RunResult) -> None:
        if run_number == 2:
            followed_up.set()

    blocked = ""
    async with Subagents(run_child, instructions="test", save_result=save_result) as agents:
        child = agents.spawn("first assignment").agent_id
        assert (await agents.wait((child,), timeout=1))[0].answer == "first"
        blocked = agents.spawn("still working").agent_id
        agents.followup(child, "second assignment")
        await asyncio.wait_for(followed_up.wait(), 1)
        statuses = await asyncio.wait_for(agents.wait((child, blocked), timeout=600), 1)
        assert statuses[0].answer == "second"
        assert statuses[1].status == "running"


@pytest.mark.parametrize("fails", [False, True], ids=["exhausted", "tool-failure"])
async def test_incomplete_child_reports_observed_work_without_extra_model_calls(fails: bool) -> None:
    async def invoke(raw: str) -> ToolResult:
        if fails:
            raise RuntimeError("save failed")
        return ToolResult(content="Saved the draft. Revenue is 515.")

    first = work_message()
    if fails:
        first = first.model_copy(update={"content": "I will save the draft now."})
    model = ScriptedModel(first)

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(FeatherAgent(model=model, tools=(tool(invoke),)))

    async with Subagents(
        run_child, instructions="test", config=SubagentConfig(max_turns=1)
    ) as agents:
        child = agents.spawn("prepare a draft").agent_id
        status = (await agents.wait((child,), timeout=1))[0]
        assert status.status == ("failed" if fails else "exhausted")
        assert "not a final answer" in status.answer
        if fails:
            assert "I will save the draft now." in status.answer
            assert "may describe intended, not completed work" in status.answer
            assert "save failed" in status.error
            assert "Saved the draft" not in status.answer
        else:
            assert "Saved the draft. Revenue is 515." in status.answer
        assert len(model.calls) == 1
        assert agents.results(child)[0].final_answer == ""
        with pytest.raises(ValueError, match="completed"):
            agents.followup(child, "do more work")


async def test_partial_report_bounds_tool_text_without_exposing_images_or_compaction() -> None:
    model = ScriptedModel(
        work_message().model_copy(update={"reasoning_content": "private reasoning"}),
        AssistantMessage(content="compaction-only summary"),
        work_message(),
        work_message(),
        tokens=120000,
    )

    async def invoke(raw: str) -> ToolResult:
        return ToolResult(content=[
            {"type": "text", "text": "Observed work: " + "x" * 10000},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
        ])

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(FeatherAgent(
            model=model,
            tools=(tool(invoke),),
            compaction=CompactionConfig(enabled=True, keep_recent_turns=0),
        ))

    async with Subagents(
        run_child, instructions="test", config=SubagentConfig(max_turns=3)
    ) as agents:
        child = agents.spawn("assignment").agent_id
        status = (await agents.wait((child,), timeout=1))[0]
        assert status.status == "exhausted"
        assert "Observed work:" in status.answer and "[truncated]" in status.answer
        assert len(status.answer) < 4500
        assert "private reasoning" not in status.answer
        assert "compaction-only summary" not in status.answer
        assert "aW1hZ2U=" not in status.answer
        result = agents.results(child)[0]
        assert result.compactions == 1
        assert "x" * 10000 in result.model_dump_json()
        assert "aW1hZ2U=" in result.model_dump_json()


async def test_wait_timeout_does_not_cancel_child_work() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def invoke(raw: str) -> ToolResult:
        started.set()
        await release.wait()
        return ToolResult(content="still alive")

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(
            FeatherAgent(
                model=ScriptedModel(work_message(), AssistantMessage(content="done")),
                tools=(tool(invoke),),
            )
        )

    async with Subagents(run_child, instructions="shared instructions") as agents:
        child = agents.spawn("assignment").agent_id
        await asyncio.wait_for(started.wait(), 2)

        assert (await agents.wait((child,), timeout=0.001))[0].status == "running"

        release.set()
        status = (await agents.wait((child,), timeout=1))[0]
        assert status.status == "completed"
        assert status.answer == "done"


async def test_long_wait_yields_to_cancellation_and_to_stop() -> None:
    started = asyncio.Event()

    async def invoke(raw: str) -> ToolResult:
        started.set()
        await asyncio.Event().wait()
        return ToolResult(content="never")

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(
            FeatherAgent(
                model=ScriptedModel(work_message(), AssistantMessage(content="done")),
                tools=(tool(invoke),),
            )
        )

    async with Subagents(run_child, instructions="shared instructions") as agents:
        child = agents.spawn("assignment").agent_id
        await asyncio.wait_for(started.wait(), 2)

        cancelled = asyncio.create_task(agents.wait((child,), timeout=600))
        await asyncio.sleep(0.01)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled, 1)
        assert agents.list()[0].status == "running"

        stopped = asyncio.create_task(agents.wait((child,), timeout=600))
        await asyncio.sleep(0.01)
        await agents.stop(child)
        status = (await asyncio.wait_for(stopped, 1))[0]
        assert status.status == "cancelled"


async def test_child_failure_preserves_its_trace_without_cancelling_sibling() -> None:
    both_started = asyncio.Event()
    fail = asyncio.Event()
    finish = asyncio.Event()
    started: set[str] = set()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        async def invoke(raw: str) -> ToolResult:
            started.add(agent_id)
            if len(started) == 2:
                both_started.set()
            if agent_id == failing:
                await fail.wait()
                raise RuntimeError("child-specific failure")
            await finish.wait()
            return ToolResult(content="sibling work")

        return await serve(
            FeatherAgent(
                model=ScriptedModel(work_message(), AssistantMessage(content="sibling answer")),
                tools=(tool(invoke),),
            )
        )

    async with Subagents(run_child, instructions="shared instructions") as agents:
        failing = agents.spawn("fail independently").agent_id
        sibling = agents.spawn("finish independently").agent_id
        await asyncio.wait_for(both_started.wait(), 2)
        fail.set()

        failed, running = await agents.wait((failing, sibling), timeout=1)

        assert failed.status == "failed"
        assert "child-specific failure" in failed.error
        assert agents.results(failing)[0].stop_reason == "failed"
        assert running.status == "running"
        finish.set()
        assert (await agents.wait((sibling,), timeout=1))[0].answer == "sibling answer"


async def test_running_children_reject_followups_and_other_parents_cannot_access_them() -> None:
    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        await asyncio.Future()
        raise AssertionError("unreachable")

    async with (
        Subagents(run_child, instructions="first parent") as first,
        Subagents(run_child, instructions="second parent") as second,
    ):
        child = first.spawn("assignment").agent_id

        with pytest.raises(ValueError, match="completed"):
            first.followup(child, "do not interrupt")
        with pytest.raises(ValueError, match="Unknown agent"):
            second.followup(child, "foreign assignment")
        with pytest.raises(ValueError, match="Unknown agent"):
            await second.wait((child,), timeout=0)
        with pytest.raises(ValueError, match="Unknown agent"):
            await second.stop(child)

        assert first.list()[0].status == "running"


async def test_child_with_delegation_tools_fails_before_its_first_model_call() -> None:
    model = ScriptedModel(AssistantMessage(content="must not run"))

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(FeatherAgent(model=model, tools=subagent_tools(agents)))

    async with Subagents(run_child, instructions="shared instructions") as agents:
        child = agents.spawn("nested delegation is forbidden").agent_id
        status = (await agents.wait((child,), timeout=1))[0]

        assert status.status == "failed"
        assert "must not receive delegation tools" in status.error
        assert model.calls == []


@pytest.mark.parametrize("completes", [False, True], ids=["exhausted", "completed-at-limit"])
async def test_turn_budget_is_shared_across_followups(completes: bool) -> None:
    model = ScriptedModel(
        AssistantMessage(content="first answer"),
        work_message(),
        AssistantMessage(content="last answer") if completes else work_message(),
    )

    async def invoke(raw: str) -> ToolResult:
        return ToolResult(content="work")

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(FeatherAgent(model=model, tools=(tool(invoke),)))

    async with Subagents(
        run_child, instructions="shared instructions", config=SubagentConfig(max_turns=3)
    ) as agents:
        child = agents.spawn("first assignment").agent_id
        assert (await agents.wait((child,), timeout=1))[0].turns == 1
        agents.followup(child, "second assignment")

        status = (await agents.wait((child,), timeout=1))[0]

        assert status.status == ("completed" if completes else "exhausted")
        assert status.turns == 3
        assert [result.num_turns for result in agents.results(child)] == [1, 2]
        with pytest.raises(ValueError, match="budget|completed"):
            agents.followup(child, "must not reset the budget")
        assert len(model.calls) == 3


async def test_followup_uses_compacted_context_without_replaying_archived_tool_output() -> None:
    model = ScriptedModel(
        work_message(),
        AssistantMessage(content="The useful finding survived compaction."),
        AssistantMessage(content="first answer"),
        AssistantMessage(content="followup answer"),
        tokens=120000,
    )

    async def invoke(raw: str) -> ToolResult:
        return ToolResult(content="archived-only tool output")

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(
            FeatherAgent(
                model=model,
                tools=(tool(invoke),),
                compaction=CompactionConfig(enabled=True, keep_recent_turns=0),
            )
        )

    async with Subagents(run_child, instructions="shared instructions") as agents:
        child = agents.spawn("original assignment").agent_id
        assert (await agents.wait((child,), timeout=1))[0].status == "completed"
        first_result = agents.results(child)[0]
        assert first_result.compactions == 1
        assert any(
            message.content == "archived-only tool output"
            for message in first_result.execution_messages()
        )
        agents.followup(child, "use the useful finding")

        assert (await agents.wait((child,), timeout=1))[0].answer == "followup answer"

        context = model.calls[-1].request["messages"]
        assert [call.kind for call in model.calls] == ["agent", "compaction", "agent", "agent"]
        assert any(
            "The useful finding survived compaction." in message["content"] for message in context
        )
        assert not any(message["role"] == "tool" for message in context)
        assert context[-2]["content"] == "first answer"
        assert context[-1] == {"role": "user", "content": "use the useful finding"}


@pytest.mark.parametrize("complete_first", [False, True], ids=["running", "completed"])
async def test_parent_exit_awaits_child_resource_cleanup(complete_first: bool) -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = asyncio.Event()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        try:
            started.set()
            if not complete_first:
                await asyncio.Future()
            return await serve(FeatherAgent(model=ScriptedModel(AssistantMessage(content="done"))))
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleaned.set()

    agents = Subagents(run_child, instructions="shared instructions")

    async def parent() -> None:
        async with agents:
            child = agents.spawn("assignment").agent_id
            await started.wait()
            if complete_first:
                assert (await agents.wait((child,), timeout=1))[0].status == "completed"

    parent_task = asyncio.create_task(parent())
    try:
        await asyncio.wait_for(cleanup_started.wait(), 2)
        assert not parent_task.done()
        assert not cleaned.is_set()
    finally:
        release_cleanup.set()
        await asyncio.wait_for(parent_task, 2)

    assert cleaned.is_set()
    assert agents.list()[0].status == ("completed" if complete_first else "cancelled")


async def test_repeated_parent_cancellation_still_awaits_child_cleanup() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = asyncio.Event()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        try:
            started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleaned.set()

    agents = Subagents(run_child, instructions="shared instructions")

    async def parent() -> None:
        async with agents:
            agents.spawn("assignment")
            await asyncio.Future()

    parent_task = asyncio.create_task(parent())
    try:
        await asyncio.wait_for(started.wait(), 2)
        parent_task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 2)
        parent_task.cancel()
        await asyncio.sleep(0)
        assert not parent_task.done()
        assert not cleaned.is_set()
    finally:
        release_cleanup.set()
        parent_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(parent_task, 2)

    assert cleaned.is_set()
    assert agents.list()[0].status == "cancelled"


async def test_stop_marks_child_cancelled_before_its_first_instruction() -> None:
    started = asyncio.Event()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        started.set()
        return await serve(FeatherAgent(model=ScriptedModel(AssistantMessage(content="too late"))))

    async with Subagents(run_child, instructions="shared instructions") as agents:
        child = agents.spawn("assignment").agent_id
        await agents.stop(child)

    status = (await agents.wait((child,), timeout=0))[0]
    assert not started.is_set()
    assert status.status == "cancelled"
    assert agents.results(child) == ()


async def test_cancelling_parent_during_explicit_stop_does_not_cancel_cleanup_twice() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = asyncio.Event()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        try:
            started.set()
            await asyncio.Future()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleaned.set()

    agents = Subagents(run_child, instructions="shared instructions")

    async def parent() -> None:
        async with agents:
            child = agents.spawn("assignment").agent_id
            await started.wait()
            await agents.stop(child)

    parent_task = asyncio.create_task(parent())
    try:
        await asyncio.wait_for(cleanup_started.wait(), 2)
        parent_task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not parent_task.done()
        assert not cleaned.is_set()
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(parent_task, 2)
    assert cleaned.is_set()


async def test_concurrency_slots_reopen_but_total_child_limit_does_not_reset() -> None:
    release = asyncio.Event()

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        await release.wait()
        return await serve(FeatherAgent(model=ScriptedModel(AssistantMessage(content="done"))))

    async with Subagents(
        run_child,
        instructions="instructions",
        config=SubagentConfig(max_concurrent=1, max_agents=2),
    ) as agents:
        first = agents.spawn("first").agent_id
        with pytest.raises(ValueError, match="concurrency"):
            agents.spawn("not admitted")
        release.set()
        assert (await agents.wait((first,), timeout=1))[0].status == "completed"
        second = agents.spawn("second").agent_id
        assert (await agents.wait((second,), timeout=1))[0].status == "completed"
        with pytest.raises(ValueError, match="total subagent"):
            agents.spawn("cannot create unlimited short-lived agents")


def _gated_children(
    releases: dict[str, asyncio.Event], answers: dict[str, list[str]]
) -> ChildRunner:
    """Children that finish each assignment when released, with scripted answers."""

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        async def invoke(raw: str) -> ToolResult:
            await releases.setdefault(agent_id, asyncio.Event()).wait()
            releases[agent_id].clear()
            return ToolResult(content="worked")

        replies = []
        for answer in answers.setdefault(agent_id, ["done", "followup done"]):
            replies += [work_message(), AssistantMessage(content=answer)]
        return await serve(FeatherAgent(model=ScriptedModel(*replies), tools=(tool(invoke),)))

    return run_child


async def test_results_the_parent_did_not_collect_are_delivered_once() -> None:
    releases: dict[str, asyncio.Event] = {}
    async with Subagents(_gated_children(releases, {}), instructions="shared") as agents:
        first = agents.spawn("first").agent_id
        second = agents.spawn("second").agent_id
        releases.setdefault(first, asyncio.Event()).set()

        await asyncio.wait_for(agents.wait_for_unreported(), 2)
        assert agents.has_unreported()
        delivered = agents.take_unreported()
        assert not agents.has_unreported()

        assert [(s.agent_id, s.status, s.answer) for s in delivered] == [
            (first, "completed", "done")
        ]
        assert agents.take_unreported() == ()
        message = render_subagent_reports(delivered)
        assert f'<report agent_id="{first}" status="completed">\ndone\n</report>' in message
        assert "not from the person" in message

        releases.setdefault(second, asyncio.Event()).set()
        assert (await agents.wait((first, second), timeout=1))[1].status == "completed"
        # The parent collected the second report itself, so nothing is left to deliver.
        assert agents.take_unreported() == ()


async def test_a_followup_result_is_delivered_even_after_the_first_was() -> None:
    releases: dict[str, asyncio.Event] = {}
    async with Subagents(_gated_children(releases, {}), instructions="shared") as agents:
        child = agents.spawn("assignment").agent_id
        releases.setdefault(child, asyncio.Event()).set()
        await asyncio.wait_for(agents.wait_for_unreported(), 2)
        assert agents.take_unreported()[0].answer == "done"

        agents.followup(child, "next assignment")
        releases[child].set()
        await asyncio.wait_for(agents.wait_for_unreported(), 2)

        assert [s.answer for s in agents.take_unreported()] == ["followup done"]


async def test_a_stop_the_parent_requested_is_not_delivered_back() -> None:
    releases: dict[str, asyncio.Event] = {}
    async with Subagents(_gated_children(releases, {}), instructions="shared") as agents:
        child = agents.spawn("assignment").agent_id
        await asyncio.sleep(0.01)

        assert (await agents.stop(child)).status == "cancelled"

        assert agents.take_unreported() == ()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agents.wait_for_unreported(), 0.05)


async def test_a_waiting_observer_does_not_receive_a_stop_the_parent_requested() -> None:
    releases: dict[str, asyncio.Event] = {}
    async with Subagents(_gated_children(releases, {}), instructions="shared") as agents:
        child = agents.spawn("assignment").agent_id
        await asyncio.sleep(0.01)
        observer = asyncio.create_task(agents.wait_for_unreported())
        await asyncio.sleep(0)

        await agents.stop(child)
        await asyncio.sleep(0.01)

        assert not observer.done()
        assert agents.take_unreported() == ()
        observer.cancel()


async def test_a_failed_report_keeps_the_reason_it_failed() -> None:
    async def failing(raw: str) -> ToolResult:
        raise RuntimeError("source.xlsx: permission denied")

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await serve(FeatherAgent(model=ScriptedModel(work_message()), tools=(tool(failing),)))

    async with Subagents(run_child, instructions="shared") as agents:
        agents.spawn("Read source.xlsx")
        await asyncio.wait_for(agents.wait_for_unreported(), 2)
        statuses = agents.take_unreported()

        assert statuses[0].status == "failed"
        assert "source.xlsx: permission denied" in render_subagent_reports(statuses)
