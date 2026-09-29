"""The final file-tool response is bounded, including errors formed on the host."""

import asyncio
import json
import re
from pathlib import Path

import pytest

from feather_harness import FileTools, LocalFileRuntime, ToolResult


async def test_large_edit_argument_error_saves_full_message(tmp_path):
    text = "é" * 30_000
    tool = FileTools(LocalFileRuntime(tmp_path)).edit_tool()
    result = await tool.invoke(
        json.dumps({"file_path": "notes.txt", "old_string": text, "new_string": text})
    )
    expected = (
        f"Invalid `new_string` parameter: {text}. old_string and new_string must be different."
    )
    saved = Path(re.search(r"Complete returned output saved at (.+?)\.\]", result.content)[1])
    try:
        assert saved.read_text() == expected
        assert len(result.content) == 24_000
        assert result.content.startswith(expected[:100])
        assert result.content.endswith(expected[-100:])
        assert result.error == "tool_error"
    finally:
        saved.unlink()


async def test_worker_path_error_saves_full_message(tmp_path):
    path = "../" + "x" * 30_000

    calls = []

    class RecordingRuntime(LocalFileRuntime):
        async def run(self, name, arguments):
            calls.append(name)
            return await super().run(name, arguments)

    runtime = RecordingRuntime(tmp_path)
    tool = FileTools(runtime).read_tool()
    result = await tool.invoke(json.dumps({"file_path": path}))
    saved = Path(re.search(r"Complete returned output saved at (.+?)\.\]", result.content)[1])
    try:
        full = saved.read_text()
        assert path in full
        assert full.startswith("Invalid `file_path` parameter:")
        assert len(result.content) == 24_000
        assert result.content.startswith(full[:100])
        assert result.content.endswith(full[-100:])
        assert result.error == "invalid_path"
        assert calls == ["read"]  # The worker saved it without another transport call.
    finally:
        saved.unlink()


async def test_transport_results_keep_revision_metadata_when_bounded(tmp_path):
    original = ToolResult(
        content="begin\n" + "x" * 30_000 + "\nend",
        metadata={"path": "notes.txt", "revision": "observed"},
    )

    class Runtime(LocalFileRuntime):
        async def run(self, name, arguments):
            if name == "read":
                return original
            return await super().run(name, arguments)

    files = FileTools(Runtime(tmp_path))
    result = await files.read_tool().invoke('{"file_path":"notes.txt"}')
    saved = Path(re.search(r"Complete returned output saved at (.+?)\.\]", result.content)[1])
    try:
        assert saved.read_text() == original.content
        assert len(result.content) == 24_000
        assert result.content.startswith("begin\n")
        assert result.content.endswith("\nend")
        assert result.error is None
        assert result.metadata == original.metadata
        assert files.revisions == {"notes.txt": "observed"}
    finally:
        saved.unlink()


async def test_spill_failure_still_returns_truncated_error():
    class UnavailableRuntime:
        async def run(self, name, arguments):
            raise OSError("storage unavailable")

    tool = FileTools(UnavailableRuntime()).edit_tool()
    text = "x" * 30_000
    result = await tool.invoke(
        json.dumps({"file_path": "notes.txt", "old_string": text, "new_string": text})
    )
    assert len(result.content) == 24_000
    assert "Output truncated" in result.content
    assert "saved at" not in result.content
    assert result.error == "tool_error"


async def test_cancellation_during_spill_propagates():
    class CancelledRuntime:
        async def run(self, name, arguments):
            raise asyncio.CancelledError

    tool = FileTools(CancelledRuntime()).edit_tool()
    text = "x" * 30_000
    with pytest.raises(asyncio.CancelledError):
        await tool.invoke(
            json.dumps({"file_path": "notes.txt", "old_string": text, "new_string": text})
        )
