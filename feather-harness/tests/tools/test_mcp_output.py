"""Aggregate MCP previews retain their text in the agent's filesystem."""

import asyncio
import base64
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from feather_harness import FileTools, ToolResult
from feather_harness.core.runner import RunResult
from feather_harness.local.filesystem import SCRIPTS
from feather_harness.scripts import _files
from feather_harness.tools.mcp import McpServerSpec, _bind, run_with_mcp

from ..mcp.server import mcp_server


class WorkerRuntime:
    def __init__(self, workspace: Path, temporary: Path):
        self.workspace = workspace
        self.temporary = temporary

    async def run(self, name, arguments):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(SCRIPTS / f"{name}_tool.py"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "TMPDIR": str(self.temporary)},
        )
        stdout, stderr = await process.communicate(
            json.dumps(
                {
                    "workspace": str(self.workspace),
                    "read_grants": [],
                    "write_grants": [],
                    "arguments": arguments,
                }
            ).encode()
        )
        assert process.returncode == 0, stderr.decode()
        return ToolResult.model_validate_json(stdout)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    workspace = tmp_path / "agent"
    temporary = tmp_path / "agent-tmp"
    coordinator = tmp_path / "coordinator-tmp"
    for path in (workspace, temporary, coordinator):
        path.mkdir()
    monkeypatch.setattr(_files.tempfile, "tempdir", str(coordinator))
    monkeypatch.setenv("TMPDIR", str(coordinator))
    return WorkerRuntime(workspace, temporary)


@dataclass
class Session:
    result: CallToolResult
    calls: int = 0

    async def call_tool(self, name, arguments):
        self.calls += 1
        return self.result


def bind(result, *, runtime=None, raise_on_error=False, images=False):
    session = Session(result)
    tool = _bind(
        McpServerSpec(
            name="test", transport="http", url="http://localhost", raise_on_error_result=raise_on_error
        ),
        session,
        Tool(name="echo", inputSchema={}),
        "mcp__test__echo",
        enable_image_tools=images,
        file_transport=runtime,
    )
    return tool, session


def text_of(result):
    return (
        result.content
        if isinstance(result.content, str)
        else "".join(block["text"] for block in result.content if block["type"] == "text")
    )


def saved_path(result):
    return Path(re.search(r"Complete returned output saved at (.+?)\.\]", text_of(result))[1])


def output_directory(runtime):
    return runtime.temporary / _files._output_directory(runtime.workspace).name


def test_unsaved_head_tail_notice_stays_byte_identical():
    assert _files.truncate_output("small") == "small"
    assert _files.truncate_output("é" * 30000) == (
        "é" * 11952
        + "\n\n...[Output truncated: kept 11952 leading and 11952 trailing characters from 30000 total.]...\n\n"
        + "é" * 11952
    )
    saved = _files.truncate_output("é" * 30000, "/tmp/workspace-output/mcp_12345678.txt")
    assert len(saved) <= 24000
    assert "long single-line text/JSON, use a shell command to slice the saved file" in saved
    assert "Complete returned output saved at /tmp/workspace-output/mcp_12345678.txt.]" in saved


@pytest.mark.parametrize("blocks", [[], ["β\n\"hello\""], ["a" * 12000, "b" * 12000]])
async def test_small_mcp_results_keep_their_exact_content(blocks, runtime):
    tool, _ = bind(
        CallToolResult(content=[TextContent(type="text", text=text) for text in blocks]),
        runtime=runtime,
    )
    result = await tool.invoke("{}")
    assert result.content == [{"type": "text", "text": text} for text in blocks]
    assert not list(runtime.temporary.iterdir())


async def test_many_small_blocks_spill_once_and_are_retrievable_in_worker(runtime):
    blocks = [f"record {index:04}: " + "é" * 90 + "\n" for index in range(300)]
    full = "".join(blocks)
    observed = []

    async def run(tools):
        result = await tools[0].invoke('{"value":"unused"}')
        observed.append(result)
        return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())

    with mcp_server(text_blocks=blocks) as (url, server):
        await run_with_mcp(
            (McpServerSpec(name="test", transport="http", url=url + "/mcp/"),),
            run,
            file_transport=runtime,
        )
        assert len(server.calls) == 1
    result = observed[0]
    preview = text_of(result)
    saved = saved_path(result)
    assert saved.parent == output_directory(runtime)
    assert stat.S_IMODE(saved.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert not list(Path(_files.tempfile.gettempdir()).iterdir())
    assert len(preview) <= 24000
    assert preview.count("[Output truncated:") == 1
    assert preview.startswith(blocks[0]) and preview.endswith(blocks[-1])

    files = FileTools(runtime)
    recovered = []
    for offset in (1, 101, 201):
        read = await files.read_tool().invoke(
            json.dumps({"file_path": str(saved), "offset": offset, "limit": 100})
        )
        assert read.error is None
        recovered.extend(re.findall(r"^\s*\d+\t(.*)$", read.content, re.MULTILINE))
    assert "\n".join(recovered) + "\n" == full
    grep = await files.grep_tool().invoke(
        json.dumps({"path": str(saved), "pattern": "record 0150"})
    )
    assert grep.error is None and "record 0150" in grep.content
    glob = await files.glob_tool().invoke(
        json.dumps({"path": str(saved.parent), "pattern": "*.txt"})
    )
    assert glob.error is None and str(saved) in glob.content
    assert len(list(saved.parent.iterdir())) == 1


async def test_mixed_images_stay_between_the_retained_text_ranges(runtime):
    first = ImageContent(type="image", data=base64.b64encode(b"first").decode(), mimeType="image/png")
    second = ImageContent(type="image", data=base64.b64encode(b"second").decode(), mimeType="image/png")
    tool, session = bind(
        CallToolResult(content=[
            TextContent(type="text", text="A" * 15000),
            first,
            TextContent(type="text", text="omitted" * 5000),
            second,
            TextContent(type="text", text="Z" * 15000),
        ]),
        runtime=runtime,
        images=True,
    )
    result = await tool.invoke("{}")
    assert [block["type"] for block in result.content] == ["text", "image_url", "image_url", "text"]
    assert result.content[0]["text"].startswith("A" * 100)
    assert "Output truncated" in result.content[0]["text"]
    assert result.content[-1]["text"].endswith("Z" * 100)
    assert [block["image_url"]["url"] for block in result.content if block["type"] == "image_url"] == [
        f"data:image/png;base64,{first.data}", f"data:image/png;base64,{second.data}"
    ]
    assert len(text_of(result)) <= 24000
    assert saved_path(result).read_text() == "A" * 15000 + "omitted" * 5000 + "Z" * 15000
    assert session.calls == 1


@pytest.mark.parametrize(
    "mode",
    ["missing-transport", "transport-failure", "worker-save-failure", "worker-error", "missing-saved-path"],
)
async def test_failed_save_preserves_full_mixed_content_without_replaying_mcp(runtime, mode):
    class UnavailableRuntime:
        async def run(self, name, arguments):
            if mode == "transport-failure":
                raise OSError("unavailable")
            return ToolResult(
                content="Complete returned output saved at /not/a/real/save.]",
                error="tool_error" if mode == "worker-error" else None,
                metadata={
                    "head_chars": 0,
                    "tail_chars": 0,
                    **({"saved_path": "/not/a/real/save"} if mode == "worker-error" else {}),
                },
            )

    if mode == "worker-save-failure":
        output_directory(runtime).write_text("not a directory")
        transport = runtime
    else:
        transport = None if mode == "missing-transport" else UnavailableRuntime()
    first = ImageContent(type="image", data=base64.b64encode(b"first").decode(), mimeType="image/png")
    second = ImageContent(type="image", data=base64.b64encode(b"second").decode(), mimeType="image/png")
    tool, session = bind(
        CallToolResult(content=[
            TextContent(type="text", text="A" * 15000), first,
            TextContent(type="text", text="omitted" * 5000), second,
            TextContent(type="text", text="Z" * 15000),
        ]),
        runtime=transport,
        images=True,
    )
    result = await tool.invoke("{}")
    assert result.content == [
        {"type": "text", "text": "A" * 15000},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{first.data}"}},
        {"type": "text", "text": "omitted" * 5000},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{second.data}"}},
        {"type": "text", "text": "Z" * 15000},
    ]
    assert result.error is None
    assert session.calls == 1


async def test_cancellation_during_spill_propagates_without_replaying():
    class CancelledRuntime:
        async def run(self, name, arguments):
            raise asyncio.CancelledError

    tool, session = bind(
        CallToolResult(content=[TextContent(type="text", text="x" * 30000)]),
        runtime=CancelledRuntime(),
    )
    with pytest.raises(asyncio.CancelledError):
        await tool.invoke("{}")
    assert session.calls == 1


@pytest.mark.parametrize("validation", [False, True])
async def test_large_mcp_errors_keep_existing_error_semantics(runtime, validation):
    message = ("Input validation error: " if validation else "database unavailable: ") + "x" * 30000
    response = CallToolResult(content=[TextContent(type="text", text=message)], isError=True)
    tool, session = bind(response, runtime=runtime, raise_on_error=True)
    if validation:
        result = await tool.invoke("{}")
        assert result.error == "invalid_arguments"
        assert len(result.content) <= 24000
        assert message in saved_path(result).read_text()
    else:
        with pytest.raises(McpError, match="database unavailable"):
            await tool.invoke("{}")
        assert not list(runtime.temporary.iterdir())
    assert session.calls == 1
    default, _ = bind(response, runtime=runtime)
    result = await default.invoke("{}")
    assert result.error is None
    assert saved_path(result).read_text() == message


async def test_image_validation_precedes_spilling(runtime):
    tool, _ = bind(
        CallToolResult(content=[
            TextContent(type="text", text="x" * 30000),
            ImageContent(type="image", data="AA==", mimeType="image/tiff"),
        ]), runtime=runtime, images=True,
    )
    with pytest.raises(ValueError, match="Unsupported MCP image MIME"):
        await tool.invoke("{}")
    assert not list(runtime.temporary.iterdir())


async def test_saved_file_grant_is_read_only_and_does_not_allow_symlink_escape(runtime):
    directory = output_directory(runtime)
    directory.mkdir()
    output = directory / "existing.txt"
    output.write_text("existing saved output")
    private = runtime.temporary / "private.txt"
    private.write_text("private")
    escape = directory / "escape.txt"
    escape.symlink_to(private)
    files = FileTools(runtime)
    read = await files.read_tool().invoke(json.dumps({"file_path": str(output)}))
    assert read.error is None and "existing saved output" in read.content
    write = await files.write_tool().invoke(json.dumps({"file_path": str(output), "content": "changed"}))
    assert write.error == "invalid_path"
    alias = runtime.workspace / "saved-output.txt"
    alias.symlink_to(output)
    write = await files.write_tool().invoke(json.dumps({"file_path": str(alias), "content": "changed"}))
    assert write.error == "invalid_path"
    assert output.read_text() == "existing saved output"
    for path in (private, escape):
        denied = await files.read_tool().invoke(json.dumps({"file_path": str(path)}))
        assert denied.error == "invalid_path"
    denied = await files.glob_tool().invoke(json.dumps({"path": str(runtime.temporary), "pattern": "*"}))
    assert denied.error == "invalid_path"


async def test_saved_output_grant_belongs_to_the_resolved_workspace(runtime):
    tool, _ = bind(
        CallToolResult(content=[TextContent(type="text", text="private output\n" * 3000)]),
        runtime=runtime,
    )
    saved = saved_path(await tool.invoke("{}"))
    other_workspace = runtime.workspace.parent / "other-agent"
    other_workspace.mkdir()
    other = FileTools(WorkerRuntime(other_workspace, runtime.temporary))
    read = await other.read_tool().invoke(json.dumps({"file_path": str(saved)}))
    assert read.error == "invalid_path"
    glob = await other.glob_tool().invoke(json.dumps({"path": str(saved.parent), "pattern": "*.txt"}))
    assert glob.error == "invalid_path"
    alias = runtime.workspace.parent / "workspace-alias"
    alias.symlink_to(runtime.workspace, target_is_directory=True)
    same = FileTools(WorkerRuntime(alias, runtime.temporary))
    read = await same.read_tool().invoke(json.dumps({"file_path": str(saved), "limit": 1}))
    assert read.error is None and "private output" in read.content


async def test_existing_read_output_spills_are_agent_readable(runtime):
    source = runtime.workspace / "source.txt"
    source.write_text("\n".join(f"line {index:04}: " + "x" * 100 for index in range(300)))
    files = FileTools(runtime)
    result = await files.read_tool().invoke(json.dumps({"file_path": str(source)}))
    saved = saved_path(result)
    result = await files.read_tool().invoke(
        json.dumps({"file_path": str(saved), "offset": 152, "limit": 1})
    )
    assert result.error is None and "line 0150:" in result.content


@pytest.mark.parametrize("loop", [False, True], ids=["outside", "symlink-loop"])
async def test_output_directory_symlink_does_not_grant_or_save_outside(runtime, loop):
    private = runtime.temporary / "private"
    private.mkdir()
    secret = private / "secret.txt"
    secret.write_text("secret")
    directory = output_directory(runtime)
    directory.symlink_to(directory if loop else private, target_is_directory=True)
    files = FileTools(runtime)
    (runtime.workspace / "ordinary.txt").write_text("ordinary workspace read")
    result = await files.read_tool().invoke('{"file_path":"ordinary.txt"}')
    assert result.error is None and "ordinary workspace read" in result.content
    result = await files.read_tool().invoke(json.dumps({"file_path": str(directory / "secret.txt")}))
    assert result.error == "invalid_path"
    tool, _ = bind(CallToolResult(content=[TextContent(type="text", text="x" * 30000)]), runtime=runtime)
    result = await tool.invoke("{}")
    assert text_of(result) == "x" * 30000
    assert list(private.iterdir()) == [secret]
