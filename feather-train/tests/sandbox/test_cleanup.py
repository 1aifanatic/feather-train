import asyncio
import json
import os
import shlex
import signal
from pathlib import Path

import pytest
from e2b import CommandExitException
from feather_harness import FileTools
from feather_harness.tools.process import ProcessTools

from feather_train import E2BRuntime


async def test_context_exit_stops_background_processes_but_keeps_sandbox(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        result = (
            await ProcessTools(transport)
            .bash_tool()
            .invoke(
                json.dumps(
                    {
                        "command": "echo $$ > /filesystem/process.pid; exec sleep 300",
                        "run_in_background": True,
                    }
                )
            )
        )
        assert "Background task: task_1" in result.content
    await sandbox.commands.run(
        f"docker exec {sandbox.container} sh -c 'test ! -f /filesystem/process.pid || ! kill -0 \"$(cat /filesystem/process.pid)\" 2>/dev/null'"
    )


async def test_cancelling_foreground_call_reaps_process_inside_container(sandbox):
    async def run():
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            await (
                ProcessTools(transport)
                .bash_tool()
                .invoke(
                    json.dumps(
                        {
                            "command": "echo $$ > /filesystem/process.pid; exec sleep 300",
                        }
                    )
                )
            )

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(10):
            while True:
                result = await sandbox.commands.run(
                    f'docker exec {sandbox.container} sh -c "cat /filesystem/process.pid 2>/dev/null || true"'
                )
                if result.stdout.strip():
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    await sandbox.commands.run(
        f"docker exec {sandbox.container} sh -c '! kill -0 \"$(cat /filesystem/process.pid)\" 2>/dev/null'"
    )


async def test_cancellation_during_launch_waits_for_container_cleanup(sandbox, monkeypatch):
    launching = asyncio.Event()
    release = asyncio.Event()
    original = sandbox.commands.run

    async def delay_launch(command, **kwargs):
        if "remote.py run " in command:
            launching.set()
            await release.wait()
        return await original(command, **kwargs)

    monkeypatch.setattr(sandbox.commands, "run", delay_launch)

    async def run():
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            await (
                ProcessTools(transport)
                .bash_tool()
                .invoke(
                    json.dumps(
                        {
                            "command": "echo $$ > /filesystem/process.pid; exec sleep 300",
                        }
                    )
                )
            )

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(10):
            await launching.wait()
    finally:
        task.cancel()
        release.set()
        async with asyncio.timeout(15):
            with pytest.raises(asyncio.CancelledError):
                await task
    await original(
        f"docker exec {sandbox.container} sh -c 'test ! -f /filesystem/process.pid || ! kill -0 \"$(cat /filesystem/process.pid)\" 2>/dev/null'"
    )


async def test_stopping_stubborn_background_process_preserves_stopped_result(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(
            json.dumps(
                {
                    "command": "trap '' TERM; echo ready > /filesystem/ready; while :; do sleep 1; done",
                    "run_in_background": True,
                }
            )
        )
        await sandbox.commands.run(
            f'docker exec {sandbox.container} sh -c "while ! test -f /filesystem/ready; do sleep .1; done"'
        )
        result = await tools.task_stop_tool().invoke(json.dumps({"task_id": "task_1"}))
        assert "State: stopped" in result.content


async def test_cancelled_upload_does_not_start_tool(sandbox, monkeypatch):
    uploaded = asyncio.Event()
    write = sandbox.files.write

    async def pause(path, data, **kwargs):
        await write(path, data, **kwargs)
        uploaded.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(sandbox.files, "write", pause)

    async def run():
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            await (
                FileTools(transport)
                .write_tool()
                .invoke(json.dumps({"file_path": "partial", "content": "x" * 2_000_000}))
            )

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(10):
            await uploaded.wait()
    finally:
        task.cancel()
        async with asyncio.timeout(15):
            with pytest.raises(asyncio.CancelledError):
                await task
    await sandbox.commands.run(f"docker exec {sandbox.container} test ! -e /filesystem/partial")
    assert not any("remote.py run " in cmd for cmd in sandbox.commands.started)


async def test_dead_container_supervisor_is_infrastructure_failure_and_children_are_reaped(sandbox):
    with pytest.raises(RuntimeError, match="without a complete result"):
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            tools = ProcessTools(transport)
            await tools.bash_tool().invoke(
                json.dumps(
                    {
                        "command": "echo once >> /filesystem/effects; echo $$ > /filesystem/child; kill -KILL $PPID; exec sleep 300",
                        "run_in_background": True,
                    }
                )
            )
            # The host collector observes docker exec ending without final status.
            await tools.task_output_tool().invoke(json.dumps({"task_id": "task_1", "block": True}))
    await sandbox.commands.run(
        f"docker exec {sandbox.container} sh -c '! kill -0 \"$(cat /filesystem/child)\" 2>/dev/null'"
    )
    result = await sandbox.commands.run(f"docker exec {sandbox.container} cat /filesystem/effects")
    assert result.stdout == "once\n"
    assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1


async def test_dead_host_collector_does_not_return_stale_running_status(sandbox):
    with pytest.raises(CommandExitException, match="supervisor died"):
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            tools = ProcessTools(transport)
            await tools.bash_tool().invoke(
                json.dumps(
                    {
                        "command": "echo once >> /filesystem/effects; echo $$ > /filesystem/child; exec sleep 300",
                        "run_in_background": True,
                    }
                )
            )
            await sandbox.commands.run(
                f'docker exec {sandbox.container} sh -c "while ! test -f /filesystem/child; do sleep .1; done"'
            )
            request = Path(
                shlex.split(
                    next(cmd for cmd in sandbox.commands.started if "remote.py run " in cmd)
                )[3]
            )
            os.kill(int(request.with_suffix(".pid").read_text()), signal.SIGKILL)
            await tools.task_output_tool().invoke(json.dumps({"task_id": "task_1", "block": True}))
    await sandbox.commands.run(
        f"docker exec {sandbox.container} sh -c '! kill -0 \"$(cat /filesystem/child)\" 2>/dev/null'"
    )
    result = await sandbox.commands.run(f"docker exec {sandbox.container} cat /filesystem/effects")
    assert result.stdout == "once\n"
    assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1


async def test_unobserved_background_infrastructure_failure_fails_context_exit(sandbox):
    with pytest.raises(CommandExitException, match="without a complete result"):
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            await (
                ProcessTools(transport)
                .bash_tool()
                .invoke(
                    json.dumps(
                        {
                            "command": "sleep .1; kill -KILL $PPID",
                            "run_in_background": True,
                        }
                    )
                )
            )
            await asyncio.sleep(0.3)


async def test_stalled_collector_is_reaped_before_metadata_is_removed(sandbox):
    with pytest.raises(CommandExitException, match="without a completed result"):
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            await (
                ProcessTools(transport)
                .bash_tool()
                .invoke(json.dumps({"command": "sleep 300", "run_in_background": True}))
            )
            request = Path(
                shlex.split(
                    next(cmd for cmd in sandbox.commands.started if "remote.py run " in cmd)
                )[3]
            )
            pid = int(request.with_suffix(".pid").read_text())
            os.kill(pid, signal.SIGSTOP)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not request.with_suffix(".result").exists()


async def test_task_stop_has_a_deadline_when_collector_is_stalled(sandbox):
    with pytest.raises(TimeoutError, match="final result"):
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as transport:
            tools = ProcessTools(transport)
            await tools.bash_tool().invoke(
                json.dumps({"command": "sleep 300", "run_in_background": True})
            )
            request = Path(
                shlex.split(
                    next(cmd for cmd in sandbox.commands.started if "remote.py run " in cmd)
                )[3]
            )
            os.kill(int(request.with_suffix(".pid").read_text()), signal.SIGSTOP)
            async with asyncio.timeout(40):
                await tools.task_stop_tool().invoke(json.dumps({"task_id": "task_1"}))
