from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from feather_harness import FileTools, LocalFileRuntime
from feather_harness.scripts import _file_worker


async def search(files, name, pattern, **arguments):
    tool = getattr(files, f"{name}_tool")()
    return await tool.invoke(json.dumps({"pattern": pattern, **arguments}))


@pytest.fixture
def files(tmp_path):
    for name, content in {
        "a.py": "needle one\nneedle two\n",
        "sub/b.ts": "needle three\n",
        "sub/c.txt": "needle four\n",
        ".hidden.py": "needle hidden\n",
        "ignored.py": "needle ignored\n",
        ".ignore": "ignored.py\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return FileTools(LocalFileRuntime(tmp_path))


async def test_glob_filters_normal_file_listing_without_overriding_ignores(files):
    result = await search(files, "glob", "**/*.{py,ts}")
    assert result.content == "a.py\nsub/b.ts"
    assert not result.error
    assert files.revisions == {}


async def test_grep_filters_files_and_limits_matches_globally(files):
    result = await search(files, "grep", "needle", include="*.{py,ts}", limit=2)
    assert result.content == (
        "a.py:1:needle one\na.py:2:needle two\n"
        "Showing 2 of at least 3 matches. Narrow the pattern, include filter, or path."
    )


@pytest.mark.parametrize(
    "pattern, paths",
    [
        ("*.py", ["root.py", "sub/nested.py"]),
        ("/*.py", ["root.py"]),
        ("!/*.py", ["sub/nested.py"]),
    ],
)
async def test_glob_and_grep_preserve_leading_slash_anchoring(tmp_path, pattern, paths):
    (tmp_path / "root.py").write_text("needle\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/nested.py").write_text("needle\n")
    files = FileTools(LocalFileRuntime(tmp_path))

    glob = await search(files, "glob", pattern)
    grep = await search(files, "grep", "needle", include=pattern)

    assert glob.content == "\n".join(paths)
    assert grep.content == "\n".join(f"{path}:1:needle" for path in paths)


async def test_grep_explicit_file_ignores_include_filter(files):
    result = await search(files, "grep", "needle", path="sub/b.ts", include="*.py")
    assert result.content == "sub/b.ts:1:needle three"


async def test_glob_limit_and_empty_scope_feedback(files):
    result = await search(files, "glob", "*", limit=1)
    assert result.content == "a.py\nShowing 1 of at least 2 files. Narrow the pattern or path."
    assert (
        await search(files, "glob", "*.py", path="sub")
    ).content == "No files matched '*.py' under sub."
    assert (await search(files, "grep", "absent", include="*.py")).content == (
        "No matches found for 'absent' in files matching '*.py' in the workspace."
    )


@pytest.mark.parametrize("name,pattern", [("grep", "["), ("glob", "["), ("grep", "bad\0pattern")])
async def test_bad_patterns_return_tool_errors(files, name, pattern):
    result = await search(files, name, pattern)
    assert result.error == "tool_error"
    assert result.content


async def test_grep_bounds_a_single_large_matching_line(tmp_path):
    (tmp_path / "large.txt").write_text("needle" + "x" * 25_000)
    result = await search(FileTools(LocalFileRuntime(tmp_path)), "grep", "needle")
    assert result.content.startswith(
        "Showing 0 of at least 1 matches within the 20,000-byte search output"
    )
    assert len(result.content) < 1000


async def test_search_rejects_escaping_roots_but_accepts_explicit_grants(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "found.txt").write_text("needle")
    (workspace / "alias").symlink_to(outside, target_is_directory=True)
    files = FileTools(LocalFileRuntime(workspace))
    for name in ("grep", "glob"):
        result = await search(files, name, "*", path="alias")
        assert result.error == "invalid_path"
        assert "`path` parameter" in result.content
    granted = FileTools(LocalFileRuntime(workspace, read_grants=(outside,)))
    result = await search(granted, "grep", "needle", path=str(outside))
    assert result.content == f"{outside}/found.txt:1:needle"
    assert (await search(granted, "glob", "*", path="../outside")).error == "invalid_path"


async def test_recursive_search_does_not_follow_nested_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("needle")
    (workspace / "alias.txt").symlink_to(outside)
    files = FileTools(LocalFileRuntime(workspace))
    assert (
        await search(files, "grep", "needle")
    ).content == "No matches found for 'needle' in the workspace."
    assert (await search(files, "glob", "*")).content == "No files matched '*' in the workspace."


async def test_search_path_kind_errors_are_model_visible(files):
    assert (await search(files, "glob", "*", path="a.py")).content.endswith(
        "The path is not a directory."
    )
    assert (await search(files, "grep", "needle", path="missing")).content.endswith(
        "The path does not exist."
    )
    assert (await search(files, "glob", "*", path="missing")).content.endswith(
        "The directory does not exist."
    )


@pytest.fixture
def slow_ripgrep(tmp_path, monkeypatch):
    binary = tmp_path / "rg"
    ready = tmp_path / "ready"
    leaked = tmp_path / "leaked"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(leaked)!r}).touch()"
    binary.write_text(
        f"#!{sys.executable}\nimport subprocess, time\nfrom pathlib import Path\n"
        f'subprocess.Popen([{sys.executable!r}, "-c", {child!r}])\n'
        f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return ready, leaked


async def test_cancelling_search_stops_its_child_process(tmp_path, slow_ripgrep):
    ready, leaked = slow_ripgrep
    task = asyncio.create_task(search(FileTools(LocalFileRuntime(tmp_path)), "grep", "needle"))
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.sleep(2.1)
    assert not leaked.exists()


@pytest.mark.parametrize("name", ["grep", "glob"])
async def test_search_timeout_returns_feedback_and_stops_children(
    tmp_path, slow_ripgrep, monkeypatch, name
):
    ready, leaked = slow_ripgrep
    monkeypatch.setattr(_file_worker, "SEARCH_TIMEOUT_SECONDS", 1)

    result = await search(FileTools(LocalFileRuntime(tmp_path)), name, "needle")

    assert ready.exists()
    assert result.error == "tool_error"
    assert result.content == f"{name} search timed out after 1 seconds."
    await asyncio.sleep(2.1)
    assert not leaked.exists()
