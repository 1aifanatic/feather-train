"""Render text-only Verifiers turns around a token sampler."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import uuid4

from renderers import (
    ParsedResponse,
    RenderedTokens,
    RendererPool,
    ToolCallParseStatus,
)
from verifiers.v1 import Client, Message, Response, SamplingConfig
from verifiers.v1.clients.train import (
    response_from_generate,
    serialize_completion,
    tool_to_wire,
)
from verifiers.v1.dialects import ChatDialect, Dialect, parse_tools
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import OverlongPromptError, ProviderError
from verifiers.v1.graph import PendingTurn

from feather_train.errors import NonRetryableHarnessError

BridgeMode = Literal["initial", "incremental", "fallback"]


@dataclass(frozen=True)
class SamplingRequest:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    stop_token_ids: list[int]


@dataclass(frozen=True)
class SampledTokens:
    completion_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    cached_input_tokens: int | None = None


class RendererClient[Sample: SampledTokens](Client, ABC):
    """Render prompts, sample tokens, and build Verifiers responses."""

    request_id_prefix = "managed"

    def __init__(
        self,
        *,
        renderer: RendererPool,
        max_context_length: int | None = None,
    ) -> None:
        self.max_context_length = max_context_length
        self.renderer = renderer

    def bridge_kwargs(self, turn: PendingTurn) -> dict[str, Any]:
        """Additional arguments required by a specialized renderer's bridge."""
        return {}

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        del headers
        if not isinstance(dialect, ChatDialect):
            raise NonRetryableHarnessError(
                "Managed renderer clients only support chat completions, got "
                f"{type(dialect).__name__}",
                status_code=400,
            )

        if turn is not None:
            # Feather returns tool results in call order, including unknown-tool errors.
            # Keep that order: Verifiers commits this same pending turn for attribution.
            prompt = turn.prompt
            tools = parse_tools(body.get("tools"))
        else:
            prompt, tools = dialect.parse_request(body)
        if any(
            isinstance(message.content, list)
            and any(part.type != "text" for part in message.content)
            for message in prompt
        ):
            raise NonRetryableHarnessError("Managed renderer clients are text-only", status_code=400)
        wire_tools = [tool_to_wire(tool) for tool in tools] if tools else None
        if wire_tools and not self.renderer.supports_tools:
            raise NonRetryableHarnessError(
                f"{self.renderer.renderer_cls.__name__} does not support tools", status_code=400
            )

        rendered, bridge_mode, bridged_turn = await self._render_prompt(
            prompt=prompt,
            turn=turn,
            tools=wire_tools,
        )
        _reject_multimodal_prompt(rendered)
        prompt_ids = list(rendered.token_ids)
        stop_token_ids = await asyncio.to_thread(self.renderer.get_stop_token_ids)
        request = _sampling_request(
            sampling_args,
            renderer_stop_token_ids=stop_token_ids,
            prompt_len=len(prompt_ids),
            max_context_length=self.max_context_length,
        )
        sampled = await self.sample_tokens(
            prompt_ids,
            request,
            session_id=session_id,
        )
        if len(sampled.completion_ids) != len(sampled.logprobs):
            raise ProviderError(
                "completion token/logprob length mismatch: "
                f"{len(sampled.completion_ids)} tokens, "
                f"{len(sampled.logprobs)} logprobs"
            )

        parsed = await asyncio.to_thread(
            self.renderer.parse_response,
            sampled.completion_ids,
            tools=wire_tools,
        )
        response_id = uuid4().hex
        parsed = _with_response_scoped_tool_ids(parsed, response_id)
        result = {
            "request_id": f"{self.request_id_prefix}-{response_id}",
            "prompt_ids": prompt_ids,
            "completion_ids": sampled.completion_ids,
            "completion_logprobs": sampled.logprobs,
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls,
            "finish_reason": _finish_reason(sampled.finish_reason, parsed),
            "multi_modal_data": rendered.multi_modal_data,
            "prompt_attribution": rendered,
        }
        response = response_from_generate(result, model, bridged_turn)
        if sampled.cached_input_tokens is not None:
            response.usage.prompt_tokens -= sampled.cached_input_tokens
            response.usage.cached_input_tokens = sampled.cached_input_tokens
        response.raw = serialize_completion(response, model)
        self.record_sample(
            turn=turn,
            prompt_ids=prompt_ids,
            sampled=sampled,
        )

        if turn is not None:
            turn.trace.info.setdefault("tito_bridge_modes", []).append(bridge_mode)
        return response

    @abstractmethod
    async def sample_tokens(
        self,
        prompt_ids: list[int],
        request: SamplingRequest,
        *,
        session_id: str | None,
    ) -> Sample:
        """Return completion tokens and their logprobs."""

    def record_sample(
        self,
        *,
        turn: PendingTurn | None,
        prompt_ids: list[int],
        sampled: Sample,
    ) -> None:
        """Record provider metadata before Verifiers appends the response to the trace."""

    async def _render_prompt(
        self,
        *,
        prompt: list[Message],
        turn: PendingTurn | None,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[RenderedTokens, BridgeMode, PendingTurn | None]:
        initial = turn is None or not turn.trace.nodes
        if turn is not None:
            tail = [message_to_wire(message) for message in turn.tail]
            previous_ids = turn.previous_token_ids() if _valid_incremental_tail(tail) else None
            if previous_ids is not None:
                previous_prompt_ids, previous_completion_ids = previous_ids
                kwargs = self.bridge_kwargs(turn)
                bridged = await asyncio.to_thread(
                    self.renderer.bridge_to_next_turn,
                    previous_prompt_ids,
                    previous_completion_ids,
                    tail,
                    tools=tools,
                    **kwargs,
                )
                if bridged is not None:
                    return bridged, "incremental", turn

        rendered = await asyncio.to_thread(
            self.renderer.render,
            [message_to_wire(message) for message in prompt],
            tools=tools,
            add_generation_prompt=True,
        )
        return rendered, ("initial" if initial else "fallback"), None


def _sampling_request(
    sampling_args: SamplingConfig,
    *,
    renderer_stop_token_ids: list[int],
    prompt_len: int,
    max_context_length: int | None,
) -> SamplingRequest:
    args = sampling_args.model_dump(exclude_none=True)
    extra_body = args.pop("extra_body", None) or {}
    args = {**extra_body, **args}
    supported = {"max_tokens", "seed", "stop", "temperature", "top_k", "top_p"}
    unsupported = sorted(set(args) - supported)
    if unsupported:
        raise NonRetryableHarnessError(
            "Managed renderer client does not support sampling arguments: "
            + ", ".join(unsupported),
            status_code=400,
        )
    max_tokens = args.get("max_tokens")
    if max_tokens is None or int(max_tokens) <= 0:
        raise NonRetryableHarnessError(
            "Managed renderer client requires max_tokens > 0", status_code=400
        )
    max_tokens = int(max_tokens)
    if max_context_length is not None:
        remaining = max_context_length - prompt_len
        if remaining <= 0:
            raise OverlongPromptError(
                f"Prompt has {prompt_len} tokens which leaves no room within "
                f"max_context_length={max_context_length}"
            )
        max_tokens = min(max_tokens, remaining)
    return SamplingRequest(
        max_tokens=max_tokens,
        temperature=float(args.get("temperature", 1.0)),
        top_p=float(args.get("top_p", 1.0)),
        top_k=int(args.get("top_k", -1)),
        seed=args.get("seed"),
        stop_token_ids=_merge_stop_token_ids(
            renderer_stop_token_ids,
            args.get("stop"),
        ),
    )


def _merge_stop_token_ids(
    renderer_stop_token_ids: list[int],
    requested_stop: Any,
) -> list[int]:
    if requested_stop is None:
        return list(renderer_stop_token_ids)
    if isinstance(requested_stop, str) or not isinstance(requested_stop, list | tuple):
        raise NonRetryableHarnessError(
            "cannot combine string stop sequences with renderer stops", status_code=400
        )
    if not all(type(token_id) is int for token_id in requested_stop):
        raise NonRetryableHarnessError(
            "cannot combine string stop sequences with renderer stops", status_code=400
        )
    return list(dict.fromkeys([*renderer_stop_token_ids, *requested_stop]))


def _valid_incremental_tail(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    roles = [message.get("role") for message in messages]
    if roles[-1] == "user":
        return all(role == "tool" for role in roles[:-1])
    return all(role == "tool" for role in roles)


def _reject_multimodal_prompt(prompt_tokens: RenderedTokens) -> None:
    multi_modal = prompt_tokens.multi_modal_data
    if multi_modal is not None and not multi_modal.is_empty():
        raise NonRetryableHarnessError("Managed renderer clients are text-only", status_code=400)


def _finish_reason(stop_reason: str, parsed: ParsedResponse) -> str:
    if stop_reason == "stop" and any(
        call.status == ToolCallParseStatus.OK for call in parsed.tool_calls
    ):
        return "tool_calls"
    return "length" if stop_reason == "length" else "stop"


def _with_response_scoped_tool_ids(
    parsed: ParsedResponse,
    response_id: str,
) -> ParsedResponse:
    calls = [
        replace(call, id=f"call_{response_id}_{index}")
        for index, call in enumerate(parsed.tool_calls)
    ]
    return replace(parsed, tool_calls=calls)
