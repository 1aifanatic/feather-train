"""Real OS write failures through the public file tools."""

import json
import subprocess
import sys


def test_partial_write_raises_without_advancing_revision(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"A" * 8192)
    # Apply the limit in a separate process so pytest and other tests are unaffected.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, json, resource, signal, sys
from pathlib import Path
from feather_harness import FileTools, LocalFileRuntime

async def main():
    files = FileTools(LocalFileRuntime(Path(sys.argv[1])))
    await files.read_tool().invoke('{"file_path":"notes.txt"}')
    revision = files.revisions['notes.txt']
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        await files.write_tool().invoke(json.dumps({'file_path': 'notes.txt', 'content': 'B' * 8192}))
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc), 'revision_unchanged': files.revisions['notes.txt'] == revision}))
    else:
        raise AssertionError('partial write reported success')

asyncio.run(main())
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    outcome = json.loads(result.stdout)
    assert "File too large" in outcome["error"]
    assert outcome["revision_unchanged"]
    assert 0 < path.stat().st_size < 8192
    assert set(path.read_bytes()) == {ord("B")}
