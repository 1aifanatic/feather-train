from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from feather_harness import FileTools, LocalFileRuntime


async def read(files, path, **kwargs):
    return await files.read_tool().invoke(json.dumps({"file_path": str(path), **kwargs}))


async def test_read_formats_requested_lines_and_records_full_file_revision(tmp_path):
    data = b"first\r\nsecond\rthird\n"
    (tmp_path / "notes.txt").write_bytes(data)
    files = FileTools(LocalFileRuntime(tmp_path))

    result = await read(files, "notes.txt", offset=2, limit=1)

    assert result.content == (
        "Contents of notes.txt:\n     2\tsecond\n"
        "Showing lines 2-2 of 3. Call read again with offset=3 to continue."
    )
    assert files.revisions["notes.txt"] == hashlib.sha256(data).hexdigest()
    assert (await read(files, tmp_path / "notes.txt", offset=2, limit=1)).content == result.content


async def test_read_decodes_utf16(tmp_path):
    (tmp_path / "wide.txt").write_bytes("résumé\n".encode("utf-16"))
    files = FileTools(LocalFileRuntime(tmp_path))

    result = await read(files, "wide.txt")
    assert "résumé" in result.content
    assert result.metadata["encoding"].startswith("utf_16")


async def test_read_empty_file_formats_one_blank_line(tmp_path):
    (tmp_path / "empty.txt").touch()
    files = FileTools(LocalFileRuntime(tmp_path))
    assert (await read(files, "empty.txt")).content == "Contents of empty.txt:\n     1\t\n"


@pytest.mark.parametrize(
    "content, expected",
    [
        ("", "The file is empty; use offset=1."),
        ("one line", "The file has 1 lines; choose an offset from 1 to 1."),
    ],
)
async def test_read_past_eof_is_a_tool_error(tmp_path, content, expected):
    (tmp_path / "notes.txt").write_text(content)
    files = FileTools(LocalFileRuntime(tmp_path))

    result = await read(files, "notes.txt", offset=2)

    assert result.error == "tool_error"
    assert expected in result.content
    assert files.revisions == {}


@pytest.mark.parametrize(
    "path, expected",
    [
        ("missing.txt", "The file does not exist."),
        ("missing/child.txt", "A parent directory does not exist."),
        ("notes.txt/child", "The parent component 'notes.txt' is not a directory."),
        (".", "The path is a directory"),
        ("../outside.txt", "Provide a path inside the workspace"),
        ("C:\\outside.txt", "Provide a path inside the workspace"),
    ],
)
async def test_invalid_paths_are_tool_results_without_recording_a_revision(
    tmp_path, path, expected
):
    (tmp_path / "notes.txt").write_text("hello")
    files = FileTools(LocalFileRuntime(tmp_path))

    result = await read(files, path)

    assert result.error is not None
    assert expected in result.content
    assert files.revisions == {}


async def test_symlinks_must_resolve_inside_workspace_or_explicit_grant(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (workspace / "alias").symlink_to(outside)
    (workspace / "notes").write_text("inside")
    (workspace / "safe").symlink_to(workspace / "notes")
    files = FileTools(LocalFileRuntime(workspace))

    assert (await read(files, "alias")).error == "invalid_path"
    assert "inside" in (await read(files, "safe")).content
    granted = FileTools(LocalFileRuntime(workspace, read_grants=(outside,)))
    assert "outside" in (await read(granted, outside)).content
    assert (await read(granted, "../outside.txt")).error == "invalid_path"
    assert "outside" in (await read(granted, "alias")).content


@pytest.mark.parametrize(
    "name, payload, expected",
    [
        ("image.png", b"anything", "Vision is not available"),
        ("document.pdf", b"anything", "binary document format"),
        ("binary.txt", b"\x00\x01\x02", "appears to be binary"),
    ],
)
async def test_non_text_files_do_not_record_editable_revisions(tmp_path, name, payload, expected):
    (tmp_path / name).write_bytes(payload)
    files = FileTools(LocalFileRuntime(tmp_path))

    assert expected in (await read(files, name)).content
    assert files.revisions == {}


async def test_large_output_is_bounded_and_full_output_is_saved(tmp_path):
    (tmp_path / "long.txt").write_text("a" * 30000 + "\nEND")
    files = FileTools(LocalFileRuntime(tmp_path))
    result = await read(files, "long.txt")
    saved = Path(re.search(r"Complete returned output saved at (.+?)\.\]", result.content)[1])
    try:
        assert len(result.content) <= 24000
        assert "Output truncated:" in result.content
        assert result.content.endswith("     2\tEND\n")
        assert (
            saved.read_text() == "Contents of long.txt:\n     1\t" + "a" * 30000 + "\n     2\tEND\n"
        )
    finally:
        saved.unlink()


async def test_read_through_a_symlinked_workspace_accepts_its_absolute_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hello")
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    files = FileTools(LocalFileRuntime(alias))
    assert (
        await read(files, alias / "notes.txt")
    ).content == "Contents of notes.txt:\n     1\thello\n"


async def test_size_limit_rejects_without_reading_or_recording_revision(tmp_path):
    with (tmp_path / "large.txt").open("wb") as handle:
        handle.truncate(10 * 1024 * 1024 + 1)
    files = FileTools(LocalFileRuntime(tmp_path))
    result = await read(files, "large.txt")
    assert "Maximum allowed size is 10MB." in result.content
    assert files.revisions == {}


async def test_colon_in_a_unix_filename_is_not_a_windows_absolute_path(tmp_path):
    (tmp_path / "C:notes.txt").write_text("hello")
    files = FileTools(LocalFileRuntime(tmp_path))
    assert (await read(files, "C:notes.txt")).content == "Contents of C:notes.txt:\n     1\thello\n"


async def test_a_grant_retargeted_to_root_is_rejected_at_execution(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hello")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    grant = tmp_path / "grant"
    grant.symlink_to(allowed, target_is_directory=True)
    files = FileTools(LocalFileRuntime(workspace, read_grants=(grant,)))
    grant.unlink()
    grant.symlink_to("/", target_is_directory=True)

    with pytest.raises(RuntimeError, match="extra path grant must not resolve to filesystem root"):
        await read(files, "notes.txt")
    assert files.revisions == {}


async def test_explicit_grant_beneath_a_file_reports_an_invalid_parent(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent_file = tmp_path / "parent"
    parent_file.write_text("hello")
    target = parent_file / "child"
    files = FileTools(LocalFileRuntime(workspace, read_grants=(target,)))

    result = await read(files, target)

    assert result.error is not None
    assert "The parent component 'parent' is not a directory." in result.content
