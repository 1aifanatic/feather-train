import json

from feather_train.sandbox import remote


def test_terminal_publication_wins_race_with_collector_exit(tmp_path, monkeypatch):
    path = tmp_path / "call.json"
    path.with_suffix(".pid").write_text("123")
    path.with_suffix(".result").write_text(
        json.dumps({"done": False, "result": {"content": "running"}})
    )
    final = {"done": True, "result": {"content": "complete"}}

    def exited(pid, request):
        path.with_suffix(".result").write_text(json.dumps(final))
        return False

    monkeypatch.setattr(remote, "alive", exited)
    assert remote.snapshot(path) == final


async def test_capture_persists_spawn_failure_for_recovery(tmp_path):
    path = tmp_path / "call.json"
    await remote.run(path, json.dumps({"capture": True, "docker": [str(tmp_path / "missing")]}))
    result = remote.snapshot(path)
    assert result["done"]
    assert result["failure"].startswith("FileNotFoundError:")
