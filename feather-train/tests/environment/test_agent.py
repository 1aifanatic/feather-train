import json

import httpx
from feather_harness import ChatCompletionsModel
from openai import AsyncOpenAI

from feather_train import E2BRuntime, run_e2b


async def test_shared_agent_loop_executes_remote_tools(sandbox):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "content": "Done."}
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "write-1",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": json.dumps(
                                {"file_path": "answer.txt", "content": "answer"}
                            ),
                        },
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        result = await run_e2b(
            "Write answer.txt",
            model=ChatCompletionsModel(client=client, model="test"),
            transport=E2BRuntime(
                sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
            ),
        )
    assert result.final_answer == "Done."
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "write-1",
        "content": "Created answer.txt.",
    }
    assert (
        await sandbox.commands.run(f"docker exec {sandbox.container} cat /filesystem/answer.txt")
    ).stdout == "answer"


async def test_tool_metrics_distinguish_schema_errors_from_invalid_paths(sandbox):
    from feather_train.agent import ToolMetrics

    turns = iter(
        [
            [
                {
                    "id": "bad-schema",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                },
                {
                    "id": "bad-path",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"file_path":"/etc/passwd"}'},
                },
                {
                    "id": "unknown",
                    "type": "function",
                    "function": {"name": "invented", "arguments": "{}"},
                },
            ],
            [],
        ]
    )
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        calls = next(turns)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done.", "tool_calls": calls},
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
            },
        )

    metrics = ToolMetrics()
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        result = await run_e2b(
            "solve",
            instructions="custom instructions",
            tools=("read", "write"),
            metrics=metrics,
            model=ChatCompletionsModel(client=client, model="test"),
            transport=E2BRuntime(
                sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
            ),
        )
    assert metrics.known_calls == 2
    assert metrics.invalid_calls == 1
    assert requests[0]["messages"][0]["content"].startswith(
        "# Agent instructions\n\ncustom instructions"
    )
    assert {tool["function"]["name"] for tool in requests[0]["tools"]} == {"read", "write"}
    assert [m["tool_call_id"] for m in requests[1]["messages"] if m["role"] == "tool"] == [
        "bad-schema",
        "bad-path",
        "unknown",
    ]
    assert result.final_answer == "Done."


async def test_web_tools_use_host_provider_and_enter_model_history(sandbox):
    from feather_harness.tools.web.exa import ExaWebProvider

    from feather_train.agent import ToolMetrics

    requests = []
    provider_requests = []

    def exa(request):
        provider_requests.append(request)
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Report",
                            "url": "https://example.com/report",
                            "highlights": ["Key finding."],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://example.com/report",
                        "url": "https://example.com/report",
                        "title": "Report",
                        "text": "# Full report",
                    }
                ],
                "statuses": [{"id": "https://example.com/report", "status": "success"}],
            },
        )

    def model(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "Done."}
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "search",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"report"}'},
                    },
                    {
                        "id": "fetch",
                        "type": "function",
                        "function": {
                            "name": "web_fetch",
                            "arguments": '{"url":"https://example.com/report"}',
                        },
                    },
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                    }
                ],
            },
        )

    metrics = ToolMetrics()
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(model))
    ) as client:
        result = await run_e2b(
            "Research the report",
            model=ChatCompletionsModel(client=client, model="test"),
            transport=E2BRuntime(
                sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
            ),
            tools=("web_search", "web_fetch"),
            metrics=metrics,
            web_provider=ExaWebProvider(
                api_key="host-only-key", transport_factory=lambda: httpx.MockTransport(exa)
            ),
        )
    assert result.final_answer == "Done."
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == [
        "web_search",
        "web_fetch",
    ]
    assert [str(request.url) for request in provider_requests] == [
        "https://api.exa.ai/search",
        "https://api.exa.ai/contents",
    ]
    messages = requests[1]["messages"][-2:]
    assert [message["tool_call_id"] for message in messages] == ["search", "fetch"]
    assert "Excerpt: Key finding." in messages[0]["content"]
    assert "# Full report" in messages[1]["content"]
    assert "host-only-key" not in json.dumps(requests)
    assert metrics.known_calls == 2
    assert metrics.invalid_calls == 0
