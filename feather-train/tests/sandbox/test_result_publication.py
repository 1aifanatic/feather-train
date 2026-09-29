import errno
import json
import sys

import pytest

from feather_train.sandbox import remote


@pytest.mark.parametrize("fault", ["heartbeat", "stdout", "result"])
async def test_publication_failure_preserves_only_successfully_saved_results(
    tmp_path, monkeypatch, fault
):
    path = tmp_path / "call.json"
    marker = tmp_path / "effects"
    final = {"done": True, "result": {"content": "complete"}}
    payload = json.dumps(
        {
            "docker": [
                sys.executable,
                "-c",
                """
import json
from pathlib import Path
import sys

request = json.load(sys.stdin)
with Path(request['marker']).open('a') as output:
    output.write('once\\n')
print(json.dumps(request['final']), flush=True)
""",
            ],
            "marker": str(marker),
            "final": final,
            "background": False,
        }
    )
    save = remote.save
    failed = False

    def fail_save(destination, value):
        nonlocal failed
        if destination.suffix == "." + fault and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "publication failed")
        save(destination, value)

    def fail_print(*args, **kwargs):
        raise BrokenPipeError("reply connection closed")

    monkeypatch.setattr(remote, "save", fail_save)
    if fault == "stdout":
        monkeypatch.setattr(remote, "print", fail_print, raising=False)

    if fault == "result":
        await remote.run(path, payload)
        recovered = await remote.inspect(path, wait=True)
        assert recovered["done"]
        assert "publication failed" in recovered["failure"]
        assert "result" not in recovered
    else:
        with pytest.raises(OSError):
            await remote.run(path, payload)
        assert await remote.inspect(path, wait=True) == final
    assert marker.read_text() == "once\n"
