import asyncio
import json
import re
import shlex
from pathlib import Path

import pytest


async def test_background_wait_is_separate_from_process_deadline(process_tools):
    started = await process_tools.bash_tool().invoke(
        json.dumps({"command": "sleep 0.3; printf finished", "run_in_background": True})
    )
    assert "Background task: task_1" in started.content
    result = await process_tools.task_output_tool().invoke(
        json.dumps({"task_id": "task_1", "timeout": 1})
    )
    assert "State: running" in result.content
    result = await process_tools.task_output_tool().invoke(
        json.dumps({"task_id": "task_1", "timeout": 2000})
    )
    assert "State: exited\nExit code: 0" in result.content
    assert result.content.endswith("Output:\nfinished")
    log = Path(re.search(r"^Log: (.+)$", result.content, re.M)[1])
    assert log.read_text() == "finished"


async def test_background_deadline_and_targeted_stop(process_tools):
    await process_tools.bash_tool().invoke(
        json.dumps({"command": "sleep 30", "run_in_background": True, "timeout": 100})
    )
    await process_tools.bash_tool().invoke(
        json.dumps(
            {"command": "sleep 30", "run_in_background": True, "description": "  other   task  "}
        )
    )
    result = await process_tools.task_output_tool().invoke(
        json.dumps({"task_id": "task_1", "timeout": 2000})
    )
    assert "State: timed_out" in result.content
    result = await process_tools.task_list_tool().invoke(json.dumps({}))
    assert result.content == "Active background tasks: 1\n- task_2 [running] other task"
    stopped = await process_tools.task_stop_tool().invoke(json.dumps({"task_id": "task_2"}))
    assert "State: stopped" in stopped.content
    assert "Exit code: 143" in stopped.content
    result = await process_tools.task_list_tool().invoke(json.dumps({}))
    assert result.content == "Active background tasks: none"
    result = await process_tools.task_list_tool().invoke(
        json.dumps({"active_only": False, "limit": 1})
    )
    assert "Showing 1 of 2 background tasks:" in result.content


async def test_inspection_flushes_background_log_before_periodic_update(process_tools, tmp_path):
    ready = tmp_path / "ready"
    await process_tools.bash_tool().invoke(
        json.dumps(
            {
                "command": f"printf fresh; touch {shlex.quote(str(ready))}; sleep 30",
                "run_in_background": True,
            }
        )
    )
    async with asyncio.timeout(5):
        while not ready.exists():
            await asyncio.sleep(0.01)
    result = await process_tools.task_output_tool().invoke(
        json.dumps({"task_id": "task_1", "block": False})
    )
    assert result.content.endswith("Output:\nfresh")
    log = Path(re.search(r"^Log: (.+)$", result.content, re.M)[1])
    assert log.read_text() == "fresh"


async def test_unknown_task_returns_tool_error(process_tools):
    result = await process_tools.task_stop_tool().invoke(json.dumps({"task_id": "missing"}))
    assert result.content == "Task missing was not found."


async def test_stopping_completed_task_preserves_its_result(process_tools):
    await process_tools.bash_tool().invoke(
        json.dumps({"command": "printf finished", "run_in_background": True})
    )
    completed = await process_tools.task_output_tool().invoke(
        json.dumps({"task_id": "task_1", "timeout": 2000})
    )
    assert "State: exited" in completed.content
    stopped = await process_tools.task_stop_tool().invoke(json.dumps({"task_id": "task_1"}))
    assert stopped.content == completed.content


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"command": "echo before\0echo after"}, id="command"),
        pytest.param({"command": "true", "workdir": "dir\0name"}, id="workdir"),
    ],
)
async def test_nul_argument_does_not_stop_worker_or_existing_background_task(
    process_tools, arguments
):
    bash = process_tools.bash_tool()
    await bash.invoke(json.dumps({"command": "sleep 60", "run_in_background": True}))

    rejected = await bash.invoke(json.dumps(arguments))
    assert rejected.error == "tool_error"
    running = await process_tools.task_output_tool().invoke(
        json.dumps({"task_id": "task_1", "block": False})
    )
    assert "State: running" in running.content
    result = await bash.invoke(json.dumps({"command": "printf still-working"}))
    assert "Exit code: 0" in result.content
    assert result.content.endswith("still-working")
