from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from feather_harness import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
    transcript_jsonl,
)
from feather_harness.core.messages import MESSAGES
from feather_harness.core.model import messages_to_wire


@pytest.mark.parametrize("content", ["result", [{"type": "text", "text": "result"}], []])
def test_text_transcript_preserves_raw_arguments_and_reasoning(content):
    messages = (
        SystemMessage(content="instructions"),
        UserMessage(content="Read the report"),
        AssistantMessage(
            reasoning_content="reason", tool_calls=(ToolCall(id="x", name="work", arguments=""),)
        ),
        ToolMessage(tool_call_id="x", content=content),
    )
    assert (
        MESSAGES.validate_python(
            [json.loads(line) for line in transcript_jsonl(messages).splitlines()]
        )
        == messages
    )
    wire = messages_to_wire(messages)
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert messages[2].tool_calls[0].arguments == ""
    assert wire[2]["reasoning_content"] == "reason"
    assert wire[1]["content"] == "Read the report"


def test_developer_messages_are_outside_the_supported_schema():
    with pytest.raises(ValidationError):
        MESSAGES.validate_python([{"role": "developer", "content": "unsupported"}])


@pytest.mark.parametrize(
    "message_type, fields",
    [
        (UserMessage, {}),
        (ToolResult, {}),
        (ToolMessage, {"tool_call_id": "x"}),
    ],
)
def test_external_image_urls_are_rejected(message_type, fields):
    with pytest.raises(ValidationError):
        message_type(
            content=[{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
            **fields,
        )
