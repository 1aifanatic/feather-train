"""Grader commands preserve their result across lost connections, without replay."""

import asyncio
import shlex
from types import SimpleNamespace

import pytest
from e2b.exceptions import AuthenticationException
from verifiers.v1 import SandboxError

from feather_train.errors import NonRetryableHarnessError
from feather_train.sandbox.lifecycle import SandboxIO


@pytest.mark.parametrize("drop", ["drop_before_reply", "drop_next_reply"])
async def test_disconnected_grader_recovers_without_repeating_execution(sandbox, drop):
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)
    setattr(sandbox.commands, drop, True)
    result = await session.exec_grader(
        "echo once >> /tmp/judge-count; sleep 0.2; printf '{\"done\":false}\\n'; "
        "printf 'judge diagnostic' >&2; exit 2",
        timeout=10,
    )
    assert (result.exit_code, result.stdout, result.stderr) == (
        2,
        '{"done":false}\n',
        "judge diagnostic",
    )
    count = await session.exec("cat", "/tmp/judge-count")
    assert count.stdout == "once\n"


async def test_grader_preserves_large_unicode_stdout_and_stderr(sandbox):
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)
    sandbox.commands.drop_next_reply = True
    content = "é" * 550_000
    result = await session.exec_grader(
        shlex.join(
            [
                "python3",
                "-c",
                "import sys; sys.stdout.write('é'*550000); sys.stderr.write('diagnostic')",
            ]
        ),
        timeout=20,
    )
    assert result.stdout == content
    assert result.stderr == "diagnostic"
    assert result.exit_code == 0


async def test_recovery_does_not_restart_execution_deadline(sandbox, monkeypatch):
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)
    run = sandbox.commands.run
    sandbox.commands.drop_before_reply = True

    async def delayed_disconnect(command, **kwargs):
        handle = await run(command, **kwargs)
        if "remote.py run " in command:
            await asyncio.sleep(1.5)
        return handle

    monkeypatch.setattr(sandbox.commands, "run", delayed_disconnect)
    # Restarting the timeout after the lost reply would let this command succeed.
    with pytest.raises(SandboxError, match="TimeoutError") as failure:
        await session.exec_grader("sleep 3; echo complete", timeout=2)
    assert isinstance(failure.value.__cause__, TimeoutError)


async def test_deadline_recovers_published_grader_result_without_replay(sandbox, monkeypatch):
    run = sandbox.commands.run

    class StalledHandle:
        def __init__(self, handle):
            self.handle = handle
            self.disconnected = asyncio.Event()

        async def wait(self):
            await self.handle.wait()
            await self.disconnected.wait()

        async def disconnect(self):
            self.disconnected.set()
            await self.handle.disconnect()

    async def stall_reply(command, **kwargs):
        if "remote.py run " in command:
            kwargs["on_stdout"] = None
            return StalledHandle(await run(command, **kwargs))
        return await run(command, **kwargs)

    monkeypatch.setattr(sandbox.commands, "run", stall_reply)
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)
    result = await session.exec_grader("echo once >> /tmp/judge-count; printf saved", timeout=2)
    assert (result.exit_code, result.stdout) == (0, "saved")
    assert (await session.exec("cat", "/tmp/judge-count")).stdout == "once\n"
    assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1
    assert any(" inspect " in cmd and cmd.endswith(" now 0") for cmd in sandbox.commands.started)


async def test_launch_timeout_does_not_wait_for_launch_confirmation():
    class StalledCommands:
        async def run(self, *args, **kwargs):
            if not kwargs.get("background"):
                return SimpleNamespace(exit_code=0, stdout="", stderr="")
            await asyncio.Event().wait()

    session = SandboxIO(SimpleNamespace(commands=StalledCommands()))
    async with asyncio.timeout(0.5):
        with pytest.raises(SandboxError):
            await session.exec_grader("true", timeout=0.02)


async def test_permanent_launch_error_only_performs_cleanup():
    class RejectedCommands:
        calls = 0

        async def run(self, *args, **kwargs):
            self.calls += 1
            if not kwargs.get("background"):
                return SimpleNamespace(exit_code=0, stdout="", stderr="")
            raise AuthenticationException("credentials rejected")

    commands = RejectedCommands()
    session = SandboxIO(SimpleNamespace(commands=commands))
    with pytest.raises(NonRetryableHarnessError, match="credentials rejected"):
        await session.exec_grader("true", timeout=1)
    assert commands.calls == 2


async def test_cancellation_disconnects_a_concurrently_returned_handle():
    caller = asyncio.current_task()

    class Handle:
        disconnected = False

        async def disconnect(self):
            self.disconnected = True

    handle = Handle()

    class Commands:
        async def run(self, *args, **kwargs):
            if not kwargs.get("background"):
                return SimpleNamespace(exit_code=0, stdout="", stderr="")
            caller.cancel()
            return handle

    session = SandboxIO(SimpleNamespace(commands=Commands()))
    with pytest.raises(asyncio.CancelledError):
        await session.exec_grader("true", timeout=1)
    assert handle.disconnected
