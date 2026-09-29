"""MCP results as seen by the next Chat Completions request."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from feather_harness import (
    ChatCompletionsModel,
    CompactionConfig,
    FeatherAgent,
    McpServerSpec,
    RunRequest,
    UserMessage,
    run_with_mcp,
)

from .server import mcp_server


@pytest.mark.parametrize("compact", [False, True], ids=["agent-request", "summary-request"])
@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        (['β\n"hello"'], [{"type": "text", "text": 'β\n"hello"'}]),
        (["one", "two"], [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]),
        ([], []),
    ],
    ids=["single-unicode-block", "multiple-blocks", "empty-result"],
)
async def test_mcp_result_preserves_pinned_chat_completions_format(blocks, expected, compact):
    requests = []
    replies = iter(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "mcp__mail__echo", "arguments": '{"value":"ignored"}'},
                    }
                ],
            },
            *([{"role": "assistant", "content": "summary"}] if compact else []),
            {"role": "assistant", "content": "done"},
        ]
    )

    async def respond(request):
        requests.append(json.loads(request.content))
        message = next(replies)
        return httpx.Response(
            200,
            json={
                "id": "response",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "usage": {
                    "prompt_tokens": 120000 if len(requests) == 1 else 10,
                    "completion_tokens": 10,
                    "total_tokens": 120010 if len(requests) == 1 else 20,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                    }
                ],
            },
        )

    with mcp_server(text_blocks=blocks) as (url, _):
        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:

            async def run(tools):
                agent = FeatherAgent(
                    model=ChatCompletionsModel(client=client, model="test"),
                    tools=tools,
                    compaction=CompactionConfig(enabled=compact, keep_recent_turns=0),
                )
                return await agent.run(RunRequest(messages=(UserMessage(content="Use echo."),)))

            result = await run_with_mcp(
                (McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run
            )

    assert result.compactions == int(compact)
    tool_messages = [m for m in requests[1]["messages"] if m["role"] == "tool"]
    assert tool_messages == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": expected,
        }
    ]
