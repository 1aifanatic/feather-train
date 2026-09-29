from __future__ import annotations

import json

import pytest

from feather_harness import (
    AssistantMessage,
    CompactionConfig,
    FeatherAgent,
    TodoList,
    ToolCall,
)

from ..helpers import ScriptedModel, request


def update(status):
    return json.dumps({"todos": [{"content": "Export PDF", "status": status}]})


async def test_read_replace_and_clear_have_distinct_state():
    todos = TodoList()
    tool = todos.tool()

    assert (await tool.invoke("{}")).content == '{"todos":[]}'
    assert todos.compaction_state() == ()
    assert (await tool.invoke(update("pending"))).content == "Todo list updated."
    assert json.loads((await tool.invoke("{}")).content) == json.loads(update("pending"))

    saved_snapshot = todos.compaction_state()
    await tool.invoke(update("completed"))
    assert "pending" in saved_snapshot[0]
    assert "completed" in todos.compaction_state()[0]

    await tool.invoke('{"todos":[]}')
    assert (await tool.invoke("{}")).content == '{"todos":[]}'
    assert todos.compaction_state() == ('## Current Todo List\n\n{"todos":[]}',)


@pytest.mark.parametrize(
    "raw",
    [
        "{bad",
        "[]",
        '{"todos":null}',
        '{"unknown":1}',
        '{"todos":[{"content":"Export PDF","status":"done"}]}',
        '{"todos":[{"content":"","status":"pending"}]}',
    ],
)
async def test_invalid_update_returns_feedback_without_losing_existing_list(raw):
    todos = TodoList()
    tool = todos.tool()
    await tool.invoke(update("pending"))
    before = todos.compaction_state()

    result = await tool.invoke(raw)

    assert result.error == "invalid_arguments"
    assert result.content.startswith("Invalid tool arguments:\n")
    assert result.content.endswith("Re-send the call with arguments that match the tool's schema.")
    assert todos.compaction_state() == before


async def test_todo_snapshot_in_compaction_precedes_the_retained_update():
    todos = TodoList()
    tool = todos.tool()
    await tool.invoke(update("pending"))
    model = ScriptedModel(
        AssistantMessage(
            tool_calls=(ToolCall(id="done", name="todo_write", arguments=update("completed")),)
        ),
        AssistantMessage(content="Work summary"),
        AssistantMessage(content="Finished"),
        tokens=120000,
    )

    result = await FeatherAgent(
        model=model,
        tools=(tool,),
        compaction=CompactionConfig(enabled=True, keep_recent_turns=1),
        compaction_tool_state=todos.compaction_state,
    ).run(request())

    messages = model.calls[-1].request["messages"]
    assert '"status":"pending"' in messages[1]["content"]
    assert json.loads(messages[2]["tool_calls"][0]["arguments"]) == json.loads(update("completed"))
    assert '"status":"completed"' in todos.compaction_state()[0]
    assert result.compactions == 1


async def test_large_validation_error_is_truncated_without_changing_todos():
    todos = TodoList()
    tool = todos.tool()
    await tool.invoke(update("pending"))
    before = await tool.invoke("{}")

    result = await tool.invoke(json.dumps({"unexpected_" + "x" * 30_000: 1}))

    assert result.error == "invalid_arguments"
    assert len(result.content) == 24_000
    assert result.content.startswith("Invalid tool arguments:")
    assert "Output truncated" in result.content
    assert "saved at" not in result.content
    assert result.content.endswith("Re-send the call with arguments that match the tool's schema.")
    assert await tool.invoke("{}") == before
