import asyncio
from contextlib import asynccontextmanager

import pytest
import verifiers.v1 as vf
from aiohttp import web
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.errors import ProviderError

from feather_train.eval import EvalConfig, EvalModelClient


@asynccontextmanager
async def provider(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0]
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        await runner.cleanup()


def config(tmp_path, url, **kwargs):
    return EvalConfig(
        hf_dataset="unused/test",
        hf_revision="a" * 40,
        model="test",
        base_url=url,
        output_dir=tmp_path / "eval",
        braintrust_project="test",
        **kwargs,
    )


def completion():
    return web.json_response(
        {
            "id": "response",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


async def request(client):
    return await client.get_response(
        ChatDialect(),
        {"messages": [{"role": "user", "content": "solve"}]},
        "test",
        vf.Sampling(),
        session_id="rollout-a",
    )


async def test_connection_pool_admits_more_than_verifiers_default_128_requests(tmp_path):
    concurrency = 129
    received = 0
    all_received = asyncio.Event()

    async def respond(req):
        nonlocal received
        received += 1
        if received == concurrency:
            all_received.set()
        await all_received.wait()
        return completion()

    async with provider(respond) as url:
        client = EvalModelClient(config(tmp_path, url, concurrency=concurrency), "test")
        try:
            async with asyncio.timeout(10):
                results = await asyncio.gather(*(request(client) for _ in range(concurrency)))
            assert len(results) == concurrency
        finally:
            all_received.set()
            await client.close()


async def test_stalled_provider_becomes_retryable_model_timeout(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(req):
        started.set()
        await release.wait()
        return completion()

    async with provider(respond) as url:
        client = EvalModelClient(config(tmp_path, url, model_request_timeout_seconds=0.2), "test")
        try:
            with pytest.raises(ProviderError) as error:
                await request(client)
            assert started.is_set()
            assert error.value.status_code == 504
        finally:
            release.set()
            await client.close()
