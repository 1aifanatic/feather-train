from __future__ import annotations

import json

import httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI

from feather_harness import (
    AgentRunError,
    ChatCompletionsModel,
    FeatherAgent,
    ModelSettings,
    RunRequest,
    Tool,
    ToolMessage,
    ToolResult,
    ToolSpec,
    UserMessage,
)


async def test_chat_settings_reasoning_alias_and_provider_response():
    requests = []

    async def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done", "reasoning": "alias"},
                        "finish_reason": "stop",
                        "logprobs": {"content": []},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        result = await ChatCompletionsModel(client=client, model="model").complete(
            [UserMessage(content="hello")],
            settings=ModelSettings(
                temperature=0, parallel_tool_calls=False, top_logprobs=0, extra_body={"seed": 7}
            ),
        )
    assert requests[0]["temperature"] == 0
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["logprobs"] is True
    assert requests[0]["seed"] == 7
    assert not (set(requests[0]) & {"stream", "tool_choice", "tools"})
    assert result.assistant.reasoning_content == "alias"
    assert requests[0]["top_logprobs"] == 0
    assert (result.input_tokens, result.output_tokens) == (12, 3)


@pytest.mark.parametrize(
    "parallel, with_tools, expected",
    [
        (None, True, None),
        (False, False, False),
        (True, False, None),
        (True, True, True),
    ],
)
async def test_parallel_tool_option_only_changes_the_model_request(parallel, with_tools, expected):
    requests = []

    async def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        await ChatCompletionsModel(client=client, model="model").complete(
            [UserMessage(content="task")],
            tools=(ToolSpec(name="work", description="work", parameters={}),) if with_tools else (),
            settings=ModelSettings(parallel_tool_calls=parallel),
        )
    wire = requests[0]
    if expected is None:
        assert "parallel_tool_calls" not in wire
    else:
        assert wire["parallel_tool_calls"] is expected
    assert ("tools" in wire) == with_tools
    assert all(spec["type"] == "function" for spec in wire.get("tools", []))
    assert all(spec["function"]["strict"] is False for spec in wire.get("tools", []))


@pytest.mark.parametrize("failure", ["connection", "empty_choices", "provider_error"])
async def test_provider_failure_preserves_completed_tool_work_without_replaying_it(failure):
    requests = []
    edits = []
    connection_error = httpx.ConnectError("provider unavailable")

    async def edit(arguments):
        edits.append(arguments)
        return ToolResult(content="File updated.")

    async def respond(req):
        requests.append(json.loads(req.content))
        if len(requests) > 1:
            if failure == "connection":
                raise connection_error
            payload = {"choices": []}
            if failure == "provider_error":
                payload = {"choices": None, "error": {"message": "provider unavailable"}}
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "edit-1",
                                    "type": "function",
                                    "function": {
                                        "name": "edit",
                                        "arguments": '{"file":"notes.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        agent = FeatherAgent(
            model=ChatCompletionsModel(client=client, model="model"),
            tools=(
                Tool(
                    spec=ToolSpec(name="edit", description="Edit a file", parameters={}),
                    invoke=edit,
                ),
            ),
        )
        with pytest.raises(AgentRunError) as raised:
            await agent.run(RunRequest(messages=(UserMessage(content="Update notes.txt"),)))

    result = raised.value.result
    if failure == "connection":
        assert isinstance(raised.value.cause, APIConnectionError)
        assert raised.value.cause.__cause__ is connection_error
    else:
        assert isinstance(raised.value.cause, RuntimeError)
        assert "response has no choices" in str(raised.value.cause)
        if failure == "provider_error":
            assert "provider unavailable" in str(raised.value.cause)
    assert edits == ['{"file":"notes.txt"}']
    assert len(requests) == 2
    assert result.stop_reason == "failed"
    assert result.num_turns == 1
    assert len(result.execution_messages()) == 3
    assert result.execution_messages()[-1] == ToolMessage(
        tool_call_id="edit-1", content="File updated."
    )
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "edit-1",
        "content": "File updated.",
    }
