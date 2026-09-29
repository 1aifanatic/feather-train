"""Argument-only errors are handled without starting a local or remote worker."""

import json

import pytest

from feather_harness import FileTools, ProcessTools, ToolResult


class RecordingTransport:
    def __init__(self):
        self.calls = []

    async def run(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolResult(content="executed")


@pytest.mark.parametrize(
    "tools, name, arguments, feedback",
    [
        (
            FileTools,
            "edit",
            {"file_path": "notes.txt", "old_string": "same", "new_string": "same"},
            "old_string and new_string must be different.",
        ),
        pytest.param(
            ProcessTools,
            "bash",
            {"command": "echo before\0echo after"},
            "The command must not contain NUL bytes.",
            id="nul-foreground",
        ),
        pytest.param(
            ProcessTools,
            "bash",
            {"command": "echo before\0echo after", "run_in_background": True},
            "The command must not contain NUL bytes.",
            id="nul-background",
        ),
        pytest.param(
            ProcessTools,
            "bash",
            {"command": "true", "workdir": "dir\0name"},
            "The workdir must not contain NUL bytes.",
            id="nul-workdir-foreground",
        ),
        pytest.param(
            ProcessTools,
            "bash",
            {"command": "true", "workdir": "dir\0name", "run_in_background": True},
            "The workdir must not contain NUL bytes.",
            id="nul-workdir-background",
        ),
        (ProcessTools, "bash", {"command": "sleep 30 &"}, "set run_in_background=true instead."),
        (
            ProcessTools,
            "bash",
            {"command": "sleep 30 & # start worker"},
            "set run_in_background=true instead.",
        ),
    ],
)
async def test_argument_errors_do_not_call_transport(tools, name, arguments, feedback):
    transport = RecordingTransport()
    tool = getattr(tools(transport), f"{name}_tool")()

    result = await tool.invoke(json.dumps(arguments))

    assert result.error == "tool_error"
    assert feedback in result.content
    assert transport.calls == []


@pytest.mark.parametrize("command", ["printf '&'", r"printf \&", "true # &", "true && true"])
async def test_literal_ampersands_comments_and_and_operators_reach_transport(command):
    transport = RecordingTransport()

    result = await ProcessTools(transport).bash_tool().invoke(json.dumps({"command": command}))

    assert result.error is None
    assert len(transport.calls) == 1
    name, arguments = transport.calls[0]
    assert name == "bash"
    assert arguments["command"] == command
