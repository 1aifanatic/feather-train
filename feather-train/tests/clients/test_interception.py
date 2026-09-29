"""Client classifications must survive the HTTP boundary and SDK retries."""

import httpx
import pytest
import verifiers.v1 as vf
from openai import APIStatusError, AsyncOpenAI
from renderers import RendererPool
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.retries import RolloutRetryConfig, should_retry
from verifiers.v1.session import RolloutSession

from feather_train.clients.tinker import TinkerClient
from feather_train.errors import NonRetryableHarnessError

from .helpers import _FakeRenderer, _FakeSamplingClient


@pytest.mark.parametrize("mode", ["configuration", "authentication", "malformed", "recovered"])
async def test_client_errors_control_request_and_rollout_retries(mode):
    class Sampler(_FakeSamplingClient):
        async def sample_async(self, **kwargs):
            if mode == "authentication":
                self.error = httpx.HTTPStatusError(
                    "invalid key",
                    request=httpx.Request("POST", "https://sample.invalid"),
                    response=httpx.Response(401),
                )
            self.logprobs = (
                None
                if mode == "malformed" or mode == "recovered" and not self.requests
                else [-0.1, -0.2]
            )
            return await super().sample_async(**kwargs)

    sampler = Sampler()
    client = TinkerClient(
        sampling_client=sampler, renderer=RendererPool(lambda: _FakeRenderer([10, 20]), 1)
    )
    trace = vf.Trace(task=vf.TraceTask(type="Test", data=vf.TaskData(idx=0, prompt="hello")))
    sampling = vf.Sampling(
        max_tokens=7, **({"reasoning_effort": "high"} if mode == "configuration" else {})
    )
    session = RolloutSession(
        ctx=vf.ModelContext(model="test", client=client, sampling=sampling), trace=trace
    )
    async with InterceptionServer() as server, server.acquire(session) as (endpoint, secret):
        async with AsyncOpenAI(base_url=endpoint + "/v1", api_key=secret, max_retries=2) as sdk:
            if mode == "recovered":
                response = await sdk.chat.completions.create(
                    model="test", messages=[{"role": "user", "content": "hello"}]
                )
                assert response.choices[0].message.content == "completion"
            else:
                with pytest.raises(APIStatusError) as failure:
                    await sdk.chat.completions.create(
                        model="test", messages=[{"role": "user", "content": "hello"}]
                    )
                assert (
                    failure.value.status_code
                    == {"configuration": 400, "authentication": 401, "malformed": 502}[mode]
                )

    if mode == "recovered":
        assert session.error is None
        assert len(sampler.requests) == 2
        assert trace.num_turns == 1
        assert [call.error.type if call.error else None for call in trace.calls] == [
            "ProviderError",
            None,
        ]
    else:
        fatal = mode != "malformed"
        assert len(trace.calls) == (1 if fatal else 3)
        assert (
            len(sampler.requests) == {"configuration": 0, "authentication": 1, "malformed": 3}[mode]
        )
        assert type(session.error).__name__ == (
            "NonRetryableHarnessError" if fatal else "ProviderError"
        )
        trace.capture_error(session.error)
        assert (
            should_retry(trace, RolloutRetryConfig(exclude=["NonRetryableHarnessError"]))
            is not fatal
        )
        assert trace.num_turns == 0


def test_non_client_fatal_errors_have_no_http_status():
    assert not hasattr(NonRetryableHarnessError("invalid task"), "status_code")
