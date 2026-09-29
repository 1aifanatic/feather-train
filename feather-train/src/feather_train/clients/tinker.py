"""Tinker sampling for the public Verifiers renderer client."""

from __future__ import annotations

import httpx
import tinker
from renderers import RendererPool
from verifiers.v1.errors import ProviderError

from feather_train.errors import NonRetryableHarnessError

from .renderer import RendererClient, SampledTokens, SamplingRequest


class TinkerClient(RendererClient[SampledTokens]):
    request_id_prefix = "tinker"

    def __init__(
        self,
        *,
        sampling_client: tinker.SamplingClient,
        renderer: RendererPool,
        max_context_length: int | None = None,
    ) -> None:
        super().__init__(renderer=renderer, max_context_length=max_context_length)
        self.sampling_client = sampling_client

    async def sample_tokens(
        self,
        prompt_ids: list[int],
        request: SamplingRequest,
        *,
        session_id: str | None,
    ) -> SampledTokens:
        del session_id
        try:
            sample = await self.sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    seed=request.seed,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop=request.stop_token_ids,
                ),
            )
        except (tinker.TinkerError, httpx.HTTPError, TimeoutError) as exc:
            raise _provider_error(exc) from exc
        if len(sample.sequences) != 1:
            raise ProviderError(
                f"Tinker sampler returned {len(sample.sequences)} sequences for num_samples=1"
            )
        sequence = sample.sequences[0]
        if sequence.logprobs is None:
            raise ProviderError("Tinker sample returned no completion logprobs")
        return SampledTokens(
            completion_ids=list(sequence.tokens),
            logprobs=list(sequence.logprobs),
            finish_reason=sequence.stop_reason,
        )


def _provider_error(exc: Exception) -> ProviderError | NonRetryableHarnessError:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
    elif isinstance(exc, tinker.APIStatusError):
        status_code = exc.status_code
    elif isinstance(exc, (tinker.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        status_code = 504
    elif isinstance(exc, (tinker.APIConnectionError, httpx.RequestError)):
        status_code = 503
    else:
        status_code = 502
    if status_code in (401, 403):
        return NonRetryableHarnessError(str(exc) or type(exc).__name__, status_code=status_code)
    return ProviderError(str(exc) or type(exc).__name__, status_code=status_code)
