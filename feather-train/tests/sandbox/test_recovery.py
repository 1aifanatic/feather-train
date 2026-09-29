import asyncio
import json
import shlex
from pathlib import Path

import pytest
from feather_harness import FileTools
from feather_harness.tools.process import ProcessTools

from feather_train import E2BRuntime


async def test_saved_result_survives_heartbeat_failure_without_replaying_tool(sandbox, monkeypatch):
    run = sandbox.commands.run

    async def fail_heartbeat(command, **kwargs):
        if "remote.py run " in command:
            request = Path(shlex.split(command)[3])
            # A real filesystem failure after the collector saves the tool result.
            heartbeat = request.with_suffix(".heartbeat")
            heartbeat.mkdir()
            try:
                return await run(command, **kwargs)
            finally:
                heartbeat.rmdir()
        return await run(command, **kwargs)

    monkeypatch.setattr(sandbox.commands, "run", fail_heartbeat)
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as runtime:
        result = (
            await FileTools(runtime)
            .write_tool()
            .invoke('{"file_path":"notes.txt","content":"complete"}')
        )
        assert result.content == "Created notes.txt."
    assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1


@pytest.mark.parametrize("padding", [0, 40_000], ids=["inline", "uploaded"])
async def test_lost_reply_recovers_result_without_repeating_mutation(sandbox, padding):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        sandbox.commands.drop_next_reply = True
        result = (
            await ProcessTools(transport)
            .bash_tool()
            .invoke(
                json.dumps(
                    {
                        "command": "#"
                        + "x" * padding
                        + "\necho once >> /filesystem/count; printf recovered",
                    }
                )
            )
        )
        assert result.content.endswith("Output:\nrecovered")
    result = await sandbox.commands.run(f"docker exec {sandbox.container} cat /filesystem/count")
    assert result.stdout == "once\n"
    assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1


async def test_recovery_waits_for_confirmed_foreground_past_startup_deadline(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        sandbox.commands.drop_before_reply = True
        result = (
            await ProcessTools(transport)
            .bash_tool()
            .invoke(json.dumps({"command": "sleep 11; printf recovered"}))
        )
        assert result.content.endswith("Output:\nrecovered")


async def test_lost_task_list_reply_retries_only_the_status_query(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(
            json.dumps({"command": "sleep 300", "run_in_background": True})
        )
        sandbox.commands.drop_next_query = True
        result = await tools.task_list_tool().invoke("{}")
        assert "task_1 [running]" in result.content
        assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1
        assert sum("remote.py tasks " in cmd for cmd in sandbox.commands.started) == 2


async def test_output_read_retries_share_one_timeout_budget(sandbox, monkeypatch):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(
            json.dumps({"command": "sleep 300", "run_in_background": True})
        )
        run = sandbox.commands.run

        async def lose_inspection(command, **kwargs):
            if "remote.py inspect " in command:
                await asyncio.sleep(6)
                raise ConnectionError("status reply lost")
            return await run(command, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(sandbox.commands, "run", lose_inspection)
            # A zero-wait inspection allows 15 seconds for transport recovery,
            # shared by all retries; three independent waits would take 18.
            async with asyncio.timeout(17):
                with pytest.raises(TimeoutError):
                    await tools.task_output_tool().invoke(
                        json.dumps({"task_id": "task_1", "timeout": 0})
                    )


async def test_file_worker_dies_after_write_without_success_or_replay(sandbox):
    # Kill the actual worker after mutation, before it can serialize the result.
    inject = """
from pathlib import Path
path = Path('/opt/feather/runtime/write_tool.py')
path.write_text(path.read_text().replace(
    '    action =',
    '    import os; os._exit(91)\\n    action =',
))
"""
    await sandbox.commands.run(
        shlex.join(["docker", "exec", "-u", "root", sandbox.container, "python3", "-c", inject])
    )
    with pytest.raises(RuntimeError, match="write failed"):
        async with E2BRuntime(
            sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
        ) as runtime:
            files = FileTools(runtime)
            await files.write_tool().invoke('{"file_path":"notes.txt","content":"committed"}')
    assert "notes.txt" not in files.revisions
    result = await sandbox.commands.run(
        f"docker exec {sandbox.container} cat /filesystem/notes.txt"
    )
    assert result.stdout == "committed"
    assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 1


@pytest.mark.parametrize("drop", ["drop_before_reply", "drop_next_reply"])
async def test_transcript_save_recovers_lost_reply_without_republication(sandbox, drop):
    async with E2BRuntime(
        sandbox,
        container=sandbox.container,
        host_runtime=sandbox.host_runtime,
        log_directory="/filesystem/.feather/tasks",
    ) as runtime:
        for text in ("é" * 100000, "newer snapshot"):
            content = json.dumps({"role": "user", "content": text}, ensure_ascii=False) + "\n"
            setattr(sandbox.commands, drop, True)
            await runtime.save_transcript(content)
            output = await sandbox.commands.run(
                f"docker exec {sandbox.container} cat /filesystem/.feather/transcript.jsonl"
            )
            assert output.stdout == content
        assert sum("remote.py run " in cmd for cmd in sandbox.commands.started) == 2
