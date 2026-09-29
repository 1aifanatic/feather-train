import asyncio
import json
import shlex
import sys

import pytest

from feather_harness import LocalShellRuntime, ProcessTools


def python(command):
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(command)}"


async def test_closing_transport_stops_background_children(tmp_path):
    marker = tmp_path / "leaked"
    ready = tmp_path / "ready"
    async with LocalShellRuntime(tmp_path, log_directory=tmp_path / "logs") as transport:
        process_tools = ProcessTools(transport)
        await process_tools.bash_tool().invoke(
            json.dumps(
                {
                    "command": f"touch {shlex.quote(str(ready))}; sleep 0.5; touch {shlex.quote(str(marker))}",
                    "run_in_background": True,
                }
            )
        )
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
    await asyncio.sleep(0.6)
    assert not marker.exists()


async def test_foreground_cancellation_cleans_up_process(tmp_path):
    ready = tmp_path / "ready"
    leaked = tmp_path / "leaked"

    async def run():
        async with LocalShellRuntime(tmp_path, log_directory=tmp_path / "logs") as transport:
            return (
                await ProcessTools(transport)
                .bash_tool()
                .invoke(
                    json.dumps(
                        {
                            "command": f"touch {shlex.quote(str(ready))}; sleep 0.5; touch {shlex.quote(str(leaked))}"
                        }
                    )
                )
            )

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.sleep(0.6)
    assert not leaked.exists()


async def test_log_creation_failure_does_not_launch_command(tmp_path):
    log_directory = tmp_path / "not_a_directory"
    log_directory.write_text("existing file")
    leaked = tmp_path / "leaked"
    with pytest.raises(RuntimeError):
        async with LocalShellRuntime(tmp_path, log_directory=log_directory) as transport:
            await (
                ProcessTools(transport)
                .bash_tool()
                .invoke(
                    json.dumps(
                        {"command": f"touch {shlex.quote(str(leaked))}", "run_in_background": True}
                    )
                )
            )
    assert not leaked.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Detached process ownership uses Linux /proc")
async def test_deadline_stops_detached_descendant(process_tools, tmp_path):
    ready = tmp_path / "ready"
    leaked = tmp_path / "leaked"
    child = f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(1); Path({str(leaked)!r}).touch()"
    command = python(
        f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)"
    )
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": command, "timeout": 300})
    )
    assert ready.exists()
    assert "State: timed_out" in result.content
    await asyncio.sleep(1.1)
    assert not leaked.exists()


async def test_stop_waits_for_deadline_cleanup_already_in_progress(process_tools, tmp_path):
    terminating = tmp_path / "terminating"
    command = python(
        "import signal,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: Path({str(terminating)!r}).touch()); time.sleep(30)"
    )
    await process_tools.bash_tool().invoke(
        json.dumps({"command": command, "timeout": 300, "run_in_background": True})
    )
    async with asyncio.timeout(5):
        while not terminating.exists():
            await asyncio.sleep(0.01)
    result = await process_tools.task_stop_tool().invoke(json.dumps({"task_id": "task_1"}))
    assert "State: timed_out" in result.content


async def test_repeated_cancellation_waits_for_cleanup(tmp_path):
    ready = tmp_path / "ready"
    terminating = tmp_path / "terminating"
    leaked = tmp_path / "leaked"
    command = python(
        "import signal,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: Path({str(terminating)!r}).touch()); "
        f"Path({str(ready)!r}).touch(); time.sleep(1); Path({str(leaked)!r}).touch()"
    )

    async def run():
        async with LocalShellRuntime(tmp_path, log_directory=tmp_path / "logs") as transport:
            return (
                await ProcessTools(transport).bash_tool().invoke(json.dumps({"command": command}))
            )

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
            task.cancel()
            while not terminating.exists():
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.6)
    assert not leaked.exists()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Reproduces a detached Linux child retaining stdout"
)
async def test_unowned_child_cannot_hang_deadline_cleanup(tmp_path, capfd):
    import os
    import signal

    pid_file = tmp_path / "detached.pid"
    try:
        async with asyncio.timeout(6):
            with pytest.raises(RuntimeError, match="Process worker exited"):
                async with LocalShellRuntime(
                    tmp_path, log_directory=tmp_path / "logs"
                ) as transport:
                    await (
                        ProcessTools(transport)
                        .bash_tool()
                        .invoke(
                            json.dumps(
                                {
                                    "command": f"setsid env -i sleep 300 & echo $! > {shlex.quote(str(pid_file))}",
                                    "timeout": 200,
                                }
                            )
                        )
                    )
        assert "cleanup incomplete" in capfd.readouterr().err
    finally:
        if pid_file.exists():
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
