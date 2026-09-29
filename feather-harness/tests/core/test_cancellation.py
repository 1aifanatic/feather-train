from __future__ import annotations

import asyncio

import pytest

from feather_harness import FeatherAgent

from ..helpers import ScriptedModel, request, tool, work_message


async def test_caller_cancellation_reaches_tool_in_callers_task():
    launched = asyncio.Event()
    tasks = []

    async def invoke(raw):
        tasks.append(asyncio.current_task())
        launched.set()
        await asyncio.Future()

    run = asyncio.create_task(
        FeatherAgent(model=ScriptedModel(work_message()), tools=(tool(invoke),)).run(request())
    )
    await asyncio.wait_for(launched.wait(), 2)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert tasks == [run]


@pytest.mark.parametrize("error", [asyncio.CancelledError(), SystemExit(17)])
async def test_control_exceptions_propagate_unchanged(error):
    async def invoke(raw):
        raise error

    with pytest.raises(type(error)) as raised:
        await FeatherAgent(model=ScriptedModel(work_message()), tools=(tool(invoke),)).run(
            request()
        )
    assert raised.value is error
