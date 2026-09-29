"""Model retries must not fork the recorded conversation or repeat tools."""

# ruff: noqa: E402
import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError, InternalServerError

vf = pytest.importorskip("verifiers.v1")

from feather_harness import FeatherAgent, RunRequest, Tool, ToolSpec, UserMessage
from feather_harness.core.model import ChatCompletionsModel
from feather_harness.core.runner import AgentRunError
from feather_harness.tools import ToolResult
from verifiers.v1.clients import EvalClient
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession


def completion(*, done=False):
    message = (
        {"role": "assistant", "content": "done"}
        if done
        else {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "append-once",
                    "type": "function",
                    "function": {"name": "append", "arguments": "{}"},
                }
            ],
        }
    )
    return httpx.Response(
        200,
        json={
            "id": "response",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {"index": 0, "message": message, "finish_reason": "stop" if done else "tool_calls"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
    )


class AgentHTTPTransport(httpx.AsyncHTTPTransport):
    """Count actual SDK attempts, optionally losing a fully committed response."""

    def __init__(self, *, lose_first_response=False):
        super().__init__()
        self.lose_first_response = lose_first_response
        self.attempts = 0

    async def handle_async_request(self, request):
        self.attempts += 1
        response = await super().handle_async_request(request)
        if self.lose_first_response and self.attempts == 1:
            await response.aread()
            await response.aclose()
            raise httpx.ReadError("response connection lost", request=request)
        return response


@asynccontextmanager
async def intercepted_agent(provider, marker, *, transport=None, timeout=3):
    client = EvalClient(base_url="https://provider.invalid/v1", api_key="test")
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    trace = vf.Trace(
        task=vf.TraceTask(type="RetryTest", data=vf.TaskData(idx=0, prompt="append once"))
    )
    session = RolloutSession(
        ctx=vf.ModelContext(model="test", client=client, sampling=vf.Sampling()), trace=trace
    )

    async def append(arguments):
        with marker.open("a") as file:
            file.write("executed\n")
        return ToolResult(content="appended")

    try:
        async with InterceptionServer() as server, server.acquire(session) as (endpoint, secret):
            async with AsyncOpenAI(
                base_url=endpoint + "/v1",
                api_key=secret,
                max_retries=2,
                timeout=timeout,
                http_client=httpx.AsyncClient(transport=transport),
            ) as sdk:
                agent = FeatherAgent(
                    model=ChatCompletionsModel(client=sdk, model="test"),
                    tools=(
                        Tool(
                            ToolSpec(name="append", description="Append a marker", parameters={}),
                            append,
                        ),
                    ),
                )
                yield agent, trace
    finally:
        await client.close()


@pytest.mark.parametrize("fault", ["lost_response", "slow_response"])
async def test_retry_reuses_committed_or_inflight_response(tmp_path, fault):
    calls = 0

    async def provider(request):
        nonlocal calls
        calls += 1
        if calls == 1 and fault == "slow_response":
            # First SDK attempt times out; its retry arrives while inference is still running.
            await asyncio.sleep(0.9)
        return completion(done=calls == 2)

    marker = tmp_path / "effects"
    transport = AgentHTTPTransport(lose_first_response=fault == "lost_response")
    async with intercepted_agent(provider, marker, transport=transport, timeout=0.3) as (
        agent,
        trace,
    ):
        result = await agent.run(RunRequest(messages=(UserMessage(content="append once"),)))
    assert result.final_answer == "done"
    assert marker.read_text() == "executed\n"
    assert calls == trace.num_turns == 2
    assert len(trace.calls) == 2
    assert transport.attempts == 3


async def test_model_retry_after_tool_execution_does_not_repeat_the_tool(tmp_path):
    calls = 0

    async def provider(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            return httpx.Response(503, json={"error": {"message": "temporarily unavailable"}})
        return completion(done=calls == 3)

    marker = tmp_path / "effects"
    async with intercepted_agent(provider, marker) as (agent, trace):
        result = await agent.run(RunRequest(messages=(UserMessage(content="append once"),)))
    assert result.final_answer == "done"
    assert marker.read_text() == "executed\n"
    assert calls == 3
    assert trace.num_turns == 2


@pytest.mark.parametrize(
    ("status", "attempts", "error"), [(400, 1, BadRequestError), (503, 3, InternalServerError)]
)
async def test_terminal_model_failure_does_not_execute_a_tool(tmp_path, status, attempts, error):
    calls = 0

    async def provider(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "request failed"}})

    marker = tmp_path / "effects"
    async with intercepted_agent(provider, marker) as (agent, trace):
        with pytest.raises(AgentRunError) as failure:
            await agent.run(RunRequest(messages=(UserMessage(content="append once"),)))
    assert isinstance(failure.value.cause, error)
    assert calls == attempts
    assert trace.num_turns == 0
    assert not marker.exists()
