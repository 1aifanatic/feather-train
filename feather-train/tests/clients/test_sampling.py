from __future__ import annotations

import httpx
import pytest
import tinker
import verifiers.v1 as vf
from renderers import (
    RendererPool,
)
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.errors import OverlongPromptError, ProviderError

from feather_train.clients.tinker import TinkerClient
from feather_train.errors import NonRetryableHarnessError

from .helpers import _body, _FakeRenderer, _FakeSamplingClient, _make_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (
            tinker.APIStatusError(
                "rate limited",
                httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://example.invalid/sample"),
                ),
                {"error": "rate limited"},
            ),
            429,
        ),
        (tinker.APITimeoutError("sample request"), 504),
        (tinker.APIConnectionError("sample request"), 503),
        (tinker.TinkerError("sampling failed"), 502),
        (
            httpx.HTTPStatusError(
                "rate limited",
                request=httpx.Request("POST", "https://sample.invalid"),
                response=httpx.Response(429),
            ),
            429,
        ),
        (httpx.ConnectError("connection failed"), 503),
        (TimeoutError("sample timed out"), 504),
    ],
)
async def test_tinker_transport_errors_use_verifiers_provider_contract(
    error: Exception,
    status_code: int,
) -> None:
    sampling_client = _FakeSamplingClient(error=error)
    client = TinkerClient(
        sampling_client=sampling_client,
        renderer=RendererPool(lambda: _FakeRenderer(rendered_ids=[10, 20]), size=1),
    )

    with pytest.raises(ProviderError) as exc_info:
        await client.get_response(
            ChatDialect(),
            _body(),
            "tinker",
            vf.Sampling(max_tokens=7),
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_context_limit_caps_completion_and_fails_when_exhausted() -> None:
    client, _, sampling_client = _make_client(max_context_length=5)

    await client.get_response(
        ChatDialect(),
        _body(),
        "tinker",
        vf.Sampling(max_tokens=7),
    )
    assert sampling_client.requests[0]["sampling_params"].max_tokens == 3

    exhausted, _, exhausted_sampling = _make_client(max_context_length=2)
    with pytest.raises(OverlongPromptError):
        await exhausted.get_response(
            ChatDialect(),
            _body(),
            "tinker",
            vf.Sampling(max_tokens=7),
        )
    assert exhausted_sampling.requests == []


@pytest.mark.asyncio
async def test_supported_sampling_preserves_renderer_stops() -> None:
    client, _, sampling_client = _make_client()

    await client.get_response(
        ChatDialect(),
        _body(),
        "tinker",
        vf.Sampling.model_validate(
            {
                "max_tokens": 7,
                "top_p": 0.8,
                "seed": 17,
                "extra_body": {"top_k": 20, "stop": [77], "seed": 42},
            }
        ),
    )

    sampling = sampling_client.requests[0]["sampling_params"]
    assert sampling.top_p == 0.8
    assert sampling.top_k == 20
    assert sampling.seed == 17
    assert sampling.stop == [99, 77]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        (
            {"max_tokens": 7, "extra_body": {"stop": ["END"]}},
            "cannot combine string stop sequences",
        ),
        (
            {"max_tokens": 7, "extra_body": {"unknown": 1}},
            "does not support sampling arguments: unknown",
        ),
        (
            {"max_tokens": 7, "reasoning_effort": "high"},
            "does not support sampling arguments: reasoning_effort",
        ),
    ],
)
async def test_unsupported_sampling_fails_before_sampling(
    sampling: dict,
    message: str,
) -> None:
    client, _, sampling_client = _make_client()

    with pytest.raises(NonRetryableHarnessError, match=message):
        await client.get_response(
            ChatDialect(),
            _body(),
            "tinker",
            vf.Sampling.model_validate(sampling),
        )

    assert sampling_client.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logprobs", "message"),
    [
        (None, "Tinker sample returned no completion logprobs"),
        ([-0.25], "completion token/logprob length mismatch: 2 tokens, 1 logprobs"),
    ],
    ids=["missing", "misaligned"],
)
async def test_rejects_samples_without_aligned_logprobs(
    logprobs: list[float] | None, message: str
) -> None:
    client, _, sampling_client = _make_client()
    sampling_client.logprobs = logprobs

    with pytest.raises(ProviderError, match=message):
        await client.get_response(ChatDialect(), _body(), "tinker", vf.Sampling(max_tokens=7))


@pytest.mark.parametrize("status", [401, 403])
async def test_authentication_failure_is_fatal(status):
    error = tinker.APIStatusError(
        "access denied",
        httpx.Response(status, request=httpx.Request("POST", "https://sample.invalid")),
        {},
    )
    client = TinkerClient(
        sampling_client=_FakeSamplingClient(error=error),
        renderer=RendererPool(lambda: _FakeRenderer([10, 20]), 1),
    )
    with pytest.raises(NonRetryableHarnessError) as failure:
        await client.get_response(ChatDialect(), _body(), "test", vf.Sampling(max_tokens=7))
    assert type(failure.value) is NonRetryableHarnessError
    assert failure.value.status_code == status
    assert failure.value.__cause__ is error


@pytest.mark.parametrize("count", [0, 2])
async def test_sampler_must_return_one_sequence(count):
    class Sampler(_FakeSamplingClient):
        async def sample_async(self, **kwargs):
            sample = await super().sample_async(**kwargs)
            sample.sequences *= count
            return sample

    client = TinkerClient(
        sampling_client=Sampler(), renderer=RendererPool(lambda: _FakeRenderer([10, 20]), 1)
    )
    with pytest.raises(ProviderError, match=f"returned {count} sequences") as failure:
        await client.get_response(ChatDialect(), _body(), "test", vf.Sampling(max_tokens=7))
    assert failure.value.status_code == 502
