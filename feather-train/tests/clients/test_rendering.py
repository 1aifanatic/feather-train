from __future__ import annotations

import asyncio

import pytest
import verifiers.v1 as vf
from renderers import (
    RenderedTokens,
    RendererPool,
)
from renderers.base import MultiModalData, PlaceholderRange
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.graph import PendingTurn, prepare_turn

from feather_train.clients.tinker import TinkerClient
from feather_train.errors import NonRetryableHarnessError

from .helpers import _body, _FakeRenderer, _FakeSamplingClient, _make_client, _trace_with_prior_turn


@pytest.mark.asyncio
async def test_initial_turn_renders_prompt_and_returns_sampled_tokens() -> None:
    client, renderer, sampling_client = _make_client()
    user = vf.UserMessage(content="hello")
    task = vf.TaskData(idx=0, name="initial", prompt=[user])
    trace = vf.Trace(task=vf.TraceTask(type="Task", data=task))
    turn = prepare_turn(trace, [user])

    response = await client.get_response(
        ChatDialect(),
        _body(),
        "tinker",
        vf.Sampling(max_tokens=7, temperature=0.2),
        session_id="session-1",
        turn=turn,
    )

    assert sampling_client.requests[0]["prompt"].to_ints() == [10, 20]
    sampling = sampling_client.requests[0]["sampling_params"]
    assert sampling.max_tokens == 7
    assert sampling.temperature == 0.2
    assert sampling.top_p == 1.0
    assert sampling.top_k == -1
    assert sampling.stop == [99]
    assert renderer.render_calls[0]["add_generation_prompt"] is True
    assert response.message.content == "completion"
    assert response.tokens is not None
    assert response.tokens.prompt_ids == [10, 20]
    assert response.tokens.completion_ids == [40, 50]
    assert response.tokens.completion_logprobs == [-0.25, -0.25]
    assert response.raw is not None
    assert response.raw["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "completion"},
            "finish_reason": "stop",
        }
    ]
    assert response.raw["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert trace.info["tito_bridge_modes"] == ["initial"]


@pytest.mark.asyncio
async def test_shared_pool_is_released_while_sampling() -> None:
    started = 0
    both_sampling = asyncio.Event()

    class Sampler(_FakeSamplingClient):
        async def sample_async(self, **kwargs):
            nonlocal started
            started += 1
            if started == 2:
                both_sampling.set()
            await both_sampling.wait()
            return await super().sample_async(**kwargs)

    pool = RendererPool(lambda: _FakeRenderer(rendered_ids=[10, 20]), size=1)
    clients = [
        TinkerClient(sampling_client=Sampler(completion_ids=ids), renderer=pool)
        for ids in ([40], [50, 51])
    ]
    responses = await asyncio.wait_for(
        asyncio.gather(
            *[
                client.get_response(ChatDialect(), _body(), "tinker", vf.Sampling(max_tokens=7))
                for client in clients
            ]
        ),
        timeout=5,
    )
    assert [response.tokens.completion_ids for response in responses] == [[40], [50, 51]]


@pytest.mark.asyncio
async def test_incremental_turn_uses_trace_graph_anchor() -> None:
    client, renderer, sampling_client = _make_client()
    trace, prompt = _trace_with_prior_turn()
    turn = PendingTurn(trace=trace, prompt=prompt, prefix_node_ids=[0, 1], path_len=4)

    await client.get_response(
        ChatDialect(),
        _body(),
        "tinker",
        vf.Sampling(max_tokens=7),
        turn=turn,
    )

    assert sampling_client.requests[0]["prompt"].to_ints() == [10, 20, 21, 30, 31]
    assert renderer.bridge_calls == [
        {
            "previous_prompt_ids": [10, 20, 21],
            "previous_completion_ids": [30],
            "new_messages": [{"role": "user", "content": "next"}],
            "tools": None,
        }
    ]
    assert renderer.render_calls == []
    assert trace.info["tito_bridge_modes"] == ["incremental"]


@pytest.mark.asyncio
async def test_failed_incremental_bridge_rerenders_full_prompt() -> None:
    client, renderer, sampling_client = _make_client()
    renderer.bridge_result = None
    trace, prompt = _trace_with_prior_turn()
    turn = PendingTurn(trace=trace, prompt=prompt, prefix_node_ids=[0, 1], path_len=4)

    await client.get_response(
        ChatDialect(),
        _body(),
        "tinker",
        vf.Sampling(max_tokens=7),
        turn=turn,
    )

    assert sampling_client.requests[0]["prompt"].to_ints() == [10, 20]
    assert renderer.render_calls[0]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
    ]
    assert trace.info["tito_bridge_modes"] == ["fallback"]


@pytest.mark.asyncio
async def test_rejects_multimodal_renderer_output_before_sampling() -> None:
    renderer = _FakeRenderer(rendered_ids=[10, 20])
    multimodal = MultiModalData(
        mm_hashes={"image": ["hash-a"]},
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
    )
    renderer.render = lambda *_args, **_kwargs: RenderedTokens(
        token_ids=[10, 20], multi_modal_data=multimodal
    )
    client, _, sampling_client = _make_client(renderer=renderer)

    with pytest.raises(NonRetryableHarnessError, match="text-only"):
        await client.get_response(
            ChatDialect(),
            _body(),
            "tinker",
            vf.Sampling(max_tokens=7),
        )

    assert sampling_client.requests == []


@pytest.mark.asyncio
async def test_response_preserves_reasoning_and_content() -> None:
    client, renderer, _ = _make_client()
    renderer.parsed.reasoning_content = "Check the requested symbol."
    renderer.parsed.content = "The symbol is AAPL."

    response = await client.get_response(
        ChatDialect(), _body(), "tinker", vf.Sampling(max_tokens=7)
    )
    assert response.message.reasoning_content == "Check the requested symbol."
    assert response.message.content == "The symbol is AAPL."


async def test_failed_response_processing_does_not_record_sample_metadata():
    class Renderer(_FakeRenderer):
        def parse_response(self, token_ids, *, tools=None):
            if self.parsed is None:
                raise ValueError("response parsing failed")
            return super().parse_response(token_ids, tools=tools)

    class Client(TinkerClient):
        def record_sample(self, *, turn, prompt_ids, sampled):
            recorded.append(sampled)

    recorded = []
    renderer = Renderer([10, 20])
    parsed = renderer.parsed
    renderer.parsed = None
    client = Client(
        sampling_client=_FakeSamplingClient(), renderer=RendererPool(lambda: renderer, 1)
    )
    with pytest.raises(ValueError, match="response parsing failed"):
        await client.get_response(ChatDialect(), _body(), "test", vf.Sampling(max_tokens=7))
    assert recorded == []

    renderer.parsed = parsed
    response = await client.get_response(ChatDialect(), _body(), "test", vf.Sampling(max_tokens=7))
    assert response.raw is not None
    assert len(recorded) == 1


async def test_image_input_is_rejected_before_text_renderer_can_drop_it():
    client, renderer, sampler = _make_client()
    body = _body()
    body["messages"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]
    with pytest.raises(NonRetryableHarnessError, match="text-only"):
        await client.get_response(ChatDialect(), body, "tinker", vf.Sampling(max_tokens=7))
    assert renderer.render_calls == []
    assert sampler.requests == []
