import asyncio
import json

import pytest
from feather_harness.tools.process import ProcessTools

from feather_train import E2BRuntime


async def test_foreground_deadline_preserves_partial_output(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        result = (
            await ProcessTools(transport)
            .bash_tool()
            .invoke(json.dumps({"command": "printf started; sleep 30", "timeout": 100}))
        )
        assert "State: timed_out" in result.content
        assert result.content.endswith("Output:\nstarted")


async def test_background_task_can_be_listed_inspected_and_stopped(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(
            json.dumps({"command": "sleep 30", "run_in_background": True})
        )
        assert "task_1 [running]" in (await tools.task_list_tool().invoke("{}")).content
        assert (
            "State: running"
            in (
                await tools.task_output_tool().invoke(
                    json.dumps({"task_id": "task_1", "block": False})
                )
            ).content
        )
        assert (
            "State: stopped"
            in (await tools.task_stop_tool().invoke(json.dumps({"task_id": "task_1"}))).content
        )


async def test_bash_runs_inside_container_with_requested_environment(sandbox):
    async with E2BRuntime(
        sandbox,
        container=sandbox.container,
        host_runtime=sandbox.host_runtime,
        env={"VALUE": 'quotes " $HOME; café'},
    ) as transport:
        result = (
            await ProcessTools(transport)
            .bash_tool()
            .invoke(json.dumps({"command": 'printf "%s" "$VALUE"'}))
        )
        assert result.content.endswith('Output:\nquotes " $HOME; café')


async def test_completed_background_task_returns_output_and_exit_status(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(
            json.dumps(
                {
                    "command": "printf before; sleep .3; printf after; exit 7",
                    "run_in_background": True,
                }
            )
        )
        result = await tools.task_output_tool().invoke(
            json.dumps({"task_id": "task_1", "block": True})
        )
        assert "Exit code: 7" in result.content
        assert result.content.endswith("Output:\nbeforeafter")


async def test_zero_timeout_returns_current_background_output(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(
            json.dumps({"command": "sleep 300", "run_in_background": True})
        )
        async with asyncio.timeout(5):
            result = await tools.task_output_tool().invoke(
                json.dumps({"task_id": "task_1", "block": True, "timeout": 0})
            )
        assert "State: running" in result.content


@pytest.mark.parametrize(
    "command", ["printf fresh; sleep 30", "printf fresh; (sleep 30 >/dev/null 2>&1 &) ; true"]
)
async def test_inspection_gets_fresh_output_and_quiet_tail_reaches_log(sandbox, command):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tools = ProcessTools(transport)
        await tools.bash_tool().invoke(json.dumps({"command": command, "run_in_background": True}))
        result = await tools.task_output_tool().invoke(
            json.dumps({"task_id": "task_1", "block": False})
        )
        assert result.content.endswith("Output:\nfresh")
        log = next(
            line.removeprefix("Log: ")
            for line in result.content.splitlines()
            if line.startswith("Log: ")
        )
        saved = await sandbox.commands.run(f"docker exec {sandbox.container} cat {log}")
        assert saved.stdout == "fresh"
