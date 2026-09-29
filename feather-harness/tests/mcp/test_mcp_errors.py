from __future__ import annotations

import json

import pytest
from mcp.shared.exceptions import McpError

from feather_harness import McpServerSpec, run_with_mcp
from feather_harness.core.runner import RunResult

from .server import mcp_server


def completed():
    return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())


@pytest.mark.parametrize(
    "arguments", ["{}", "[]", "{"], ids=["missing-required", "not-an-object", "invalid-json"]
)
async def test_invalid_arguments_do_not_reach_server(arguments):
    with mcp_server() as (url, server):

        async def run(tools):
            result = await tools[0].invoke(arguments)
            assert result.error == "invalid_arguments"
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)
        assert server.calls == []


async def test_malformed_arguments_are_not_echoed():
    with mcp_server() as (url, server):

        async def run(tools):
            result = await tools[0].invoke('{"value":"' + "x" * 100_000)
            assert result.error == "invalid_arguments"
            assert result.content == (
                "Invalid tool arguments:\n"
                f"Invalid JSON input for tool {tools[0].spec.name}\n"
                "Re-send the call with arguments that match the tool's schema."
            )
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)
        assert server.calls == []


async def test_runtime_failure_is_not_replayed():
    with mcp_server() as (url, server):

        async def run(tools):
            await tools[1].invoke("{}")

        with pytest.raises(McpError, match="database unavailable"):
            await run_with_mcp(
                (
                    McpServerSpec(
                        name="mail", transport="http", url=url + "/mcp/", raise_on_error_result=True
                    ),
                ),
                run,
            )
        assert server.calls == [("failure", {})]


async def test_error_content_is_returned_by_default():
    with mcp_server() as (url, _):

        async def run(tools):
            result = await tools[1].invoke("{}")
            assert result.error is None
            assert result.content == [{"text": "database unavailable", "type": "text"}]
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)


async def test_server_validation_error_becomes_model_feedback():
    with mcp_server() as (url, _):

        async def run(tools):
            result = await tools[1].invoke(
                json.dumps({"message": "Input validation error: value must be a string"})
            )
            assert result.error == "invalid_arguments"
            assert "Input validation error: value must be a string" in result.content
            return completed()

        await run_with_mcp(
            (
                McpServerSpec(
                    name="mail", transport="http", url=url + "/mcp/", raise_on_error_result=True
                ),
            ),
            run,
        )


async def test_image_result_fails_explicitly():
    with mcp_server() as (url, _):

        async def run(tools):
            await tools[2].invoke("{}")

        with pytest.raises(ValueError, match="not currently supported"):
            await run_with_mcp(
                (McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run
            )
