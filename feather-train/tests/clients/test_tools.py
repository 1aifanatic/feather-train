from __future__ import annotations

import pytest
import verifiers.v1 as vf
from renderers import (
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
)
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.graph import prepare_turn

from feather_train.errors import NonRetryableHarnessError

from .helpers import _body, _FakeRenderer, _make_client


@pytest.mark.asyncio
async def test_tool_schema_and_strictness_reach_renderer() -> None:
    client, renderer, _ = _make_client()
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a ticker",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            "strict": True,
        },
    }

    await client.get_response(
        ChatDialect(),
        _body(tools=[tool]),
        "tinker",
        vf.Sampling(max_tokens=7),
    )

    assert renderer.render_calls[0]["tools"] == [tool]


@pytest.mark.asyncio
async def test_unsupported_tools_fail_before_rendering() -> None:
    renderer = _FakeRenderer(rendered_ids=[10, 20])
    renderer.supports_tools = False
    client, renderer, sampling_client = _make_client(renderer=renderer)

    with pytest.raises(NonRetryableHarnessError, match="does not support tools"):
        await client.get_response(
            ChatDialect(),
            _body(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a ticker",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
            "tinker",
            vf.Sampling(max_tokens=7),
        )

    assert renderer.render_calls == []
    assert sampling_client.requests == []


@pytest.mark.asyncio
async def test_response_keeps_named_malformed_calls_and_omits_unnamed_calls() -> None:
    renderer = _FakeRenderer(rendered_ids=[10, 20])
    renderer.parsed = ParsedResponse(
        content="",
        tool_calls=[
            ParsedToolCall(
                raw="raw1",
                name="lookup",
                arguments='{"query": ',
                status=ToolCallParseStatus.INVALID_JSON,
                id="call_0",
            ),
            ParsedToolCall(
                raw="raw2",
                name=None,
                status=ToolCallParseStatus.MISSING_NAME,
                id="call_1",
            ),
        ],
    )
    client, _, _ = _make_client(renderer=renderer)

    response = await client.get_response(
        ChatDialect(),
        _body(),
        "tinker",
        vf.Sampling(max_tokens=7),
    )

    assert response.finish_reason == "stop"
    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0].name == "lookup"
    assert response.message.tool_calls[0].arguments == '{"query": '


@pytest.fixture
def tool_client():
    renderer = _FakeRenderer(rendered_ids=[10, 20])
    renderer.parsed = ParsedResponse(
        content="",
        tool_calls=[
            ParsedToolCall(
                raw="raw",
                name="lookup",
                arguments='{"query": "AAPL"}',
                status=ToolCallParseStatus.OK,
                id="call_0",
            )
        ],
    )
    client, _, _ = _make_client(renderer=renderer)
    return client


@pytest.mark.asyncio
async def test_valid_tool_call_sets_tool_calls_finish_reason(tool_client) -> None:
    response = await tool_client.get_response(
        ChatDialect(), _body(), "tinker", vf.Sampling(max_tokens=7)
    )
    assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_tool_call_ids_are_unique_across_responses(tool_client) -> None:
    first = await tool_client.get_response(
        ChatDialect(), _body(), "tinker", vf.Sampling(max_tokens=7)
    )
    second = await tool_client.get_response(
        ChatDialect(), _body(), "tinker", vf.Sampling(max_tokens=7)
    )
    assert first.message.tool_calls[0].id != second.message.tool_calls[0].id


@pytest.mark.asyncio
async def test_tool_order_and_token_attribution_survive_trace_commit() -> None:
    class ContentRenderer(_FakeRenderer):
        def render(self, messages, *, tools=None, add_generation_prompt=False):
            # Character tokens make a result attributed to the wrong call visible.
            return RenderedTokens(
                token_ids=[ord(char) for message in messages for char in message["content"]],
                message_indices=[
                    index for index, message in enumerate(messages) for _ in message["content"]
                ],
                message_roles=[message["role"] for message in messages],
                is_content=[True for message in messages for _ in message["content"]],
            )

    user = vf.UserMessage(content="Check both files")
    assistant = vf.AssistantMessage(
        content="Checking",
        tool_calls=[
            vf.ToolCall(id="first", name="read", arguments='{"path":"a"}'),
            vf.ToolCall(id="unknown", name="missing_tool", arguments="{}"),
            vf.ToolCall(id="last", name="read", arguments='{"path":"b"}'),
        ],
    )
    results = [
        vf.ToolMessage(tool_call_id="first", content="First file"),
        vf.ToolMessage(tool_call_id="unknown", content="Unknown tool: missing_tool"),
        vf.ToolMessage(tool_call_id="last", content="Last file"),
    ]
    prompt = [user, assistant, *results]
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, name="tools", prompt=[user]))
    )
    turn = prepare_turn(trace, prompt)
    client, _, sampler = _make_client(renderer=ContentRenderer(rendered_ids=[]))

    response = await client.get_response(
        ChatDialect(),
        _body(messages=[message_to_wire(message) for message in prompt]),
        "tinker",
        vf.Sampling(max_tokens=7),
        turn=turn,
    )
    turn.commit(response)

    assert sampler.requests[0]["prompt"].to_ints() == [
        ord(char) for message in prompt for char in message.content
    ]
    tool_nodes = [node for node in trace.nodes if node.message.role == "tool"]
    assert [node.message for node in tool_nodes] == results
    for node, result in zip(tool_nodes, results, strict=True):
        assert node.token_ids == [ord(char) for char in result.content]
        assert node.is_content == [True] * len(result.content)
        assert node.mask == [False] * len(result.content)
