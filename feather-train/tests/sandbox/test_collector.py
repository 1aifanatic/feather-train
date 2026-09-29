import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from e2b.exceptions import AuthenticationException

from feather_train.sandbox.collector import Collector


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    "state",
    [
        {"done": True, "result": {"content": "saved"}},
        {"done": True, "failure": "worker failed"},
        {"done": False, "result": {"content": "still running"}},
    ],
)
async def test_deadline_reads_saved_state_without_relaunching(background, state):
    collector, commands, _ = _stalled_collector(state)
    deadline = asyncio.get_running_loop().time() + 0.02
    if not background and not state["done"]:
        with pytest.raises(TimeoutError):
            await collector.execute("token", background, deadline, "launch")
    else:
        assert await collector.execute("token", background, deadline, "launch") == state
    assert commands.run.await_count == 2
    assert commands.run.call_args_list[0].args == ("launch",)
    assert " inspect " in commands.run.call_args.args[0]
    assert commands.run.call_args.args[0].endswith(" now 0")
    assert commands.run.call_args.kwargs["timeout"] == 15


async def test_external_cancellation_does_not_recover_or_relaunch():
    collector, commands, started = _stalled_collector({"done": True})
    task = asyncio.create_task(
        collector.execute("token", False, asyncio.get_running_loop().time() + 60, "launch")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert commands.run.await_count == 1


async def test_deadline_recovery_preserves_permanent_provider_error():
    collector, commands, _ = _stalled_collector({"done": True})
    handle = next(commands.run.side_effect)
    commands.run.side_effect = [handle, AuthenticationException("credentials rejected")]
    with pytest.raises(AuthenticationException, match="credentials rejected"):
        await collector.execute("token", False, asyncio.get_running_loop().time() + 0.02, "launch")
    assert commands.run.await_count == 2


async def test_deadline_recovery_read_has_a_bounded_budget(monkeypatch):
    collector, commands, _ = _stalled_collector({"done": True})

    async def stalled_read(*args, **kwargs):
        await asyncio.Event().wait()

    read = AsyncMock(side_effect=stalled_read)
    monkeypatch.setattr(collector, "read", read)
    async with asyncio.timeout(17):
        with pytest.raises(TimeoutError, match="without a recoverable result"):
            await collector.execute(
                "token", False, asyncio.get_running_loop().time() + 0.02, "launch"
            )
    read.assert_awaited_once()
    assert read.call_args.kwargs["timeout"] == 15
    assert commands.run.await_count == 1


def _stalled_collector(state):
    started = asyncio.Event()
    disconnected = asyncio.Event()

    async def wait():
        started.set()
        await disconnected.wait()

    async def disconnect():
        disconnected.set()

    handle = SimpleNamespace(wait=wait, disconnect=disconnect)
    commands = SimpleNamespace(
        run=AsyncMock(side_effect=[handle, SimpleNamespace(stdout=json.dumps(state))])
    )
    return Collector(SimpleNamespace(commands=commands), "test"), commands, started
