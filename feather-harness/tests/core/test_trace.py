"""Trace segments preserve actual requests without duplicating executed actions."""

import pytest

from feather_harness import (
    AgentRunError,
    AssistantMessage,
    CompactionConfig,
    FeatherAgent,
    RunResult,
    ToolMessage,
    ToolResult,
)

from ..helpers import ScriptedModel, request, tool, work_message


async def test_segments_reconstruct_every_request_across_compaction():
    async def invoke(raw):
        return ToolResult(content="saved")

    model = ScriptedModel(
        work_message(),
        work_message(),
        AssistantMessage(content="summary"),
        work_message(),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    result = await FeatherAgent(
        model=model,
        tools=(tool(invoke),),
        compaction=CompactionConfig(enabled=True, min_turns_before_compaction=2),
    ).run(request())
    restored = RunResult.model_validate_json(result.model_dump_json())
    requests = []
    for segment in restored.segments:
        context = list(segment.input)
        for message in segment.messages:
            if isinstance(message, AssistantMessage):
                requests.append([m.model_dump(mode="json") for m in context])
            context.append(message)
    assert requests == [call.request["messages"] for call in model.calls]
    assert [s.kind for s in restored.segments] == ["execution", "compaction", "execution"]
    assert [m.content for m in restored.execution_messages() if isinstance(m, ToolMessage)] == [
        "saved"
    ] * 3
    assert "summary" not in [m.content for m in restored.execution_messages()]


async def test_failed_summary_request_is_retained_without_a_new_execution_segment():
    class FailingSummaryModel(ScriptedModel):
        async def complete(self, context, *, kind="agent", **kwargs):
            if kind == "compaction":
                self.summary_input = tuple(context)
                raise ConnectionError("summary connection lost")
            return await super().complete(context, kind=kind, **kwargs)

    async def invoke(raw):
        return ToolResult(content="saved")

    model = FailingSummaryModel(work_message(), tokens=120000)
    with pytest.raises(AgentRunError, match="summary connection lost") as error:
        await FeatherAgent(
            model=model, tools=(tool(invoke),), compaction=CompactionConfig(enabled=True)
        ).run(request())
    result = error.value.result
    assert [s.kind for s in result.segments] == ["execution", "compaction"]
    assert result.segments[-1].input == model.summary_input
    assert result.segments[-1].messages == ()
    assert result.execution_messages()[-1] == ToolMessage(tool_call_id="k", content="saved")
    assert result.compactions == 0
