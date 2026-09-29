"""Stdlib-only grader process ownership, executed inside the verifier with Python -c."""

import fcntl
import os
import signal
import sys
import time
from pathlib import Path


def run(state: str, command: str) -> None:
    pid = os.fork()
    if pid:
        _, status = os.waitpid(pid, 0)
        code = os.waitstatus_to_exitcode(status)
        raise SystemExit(code if code >= 0 else 128 - code)

    os.setsid()
    with _locked_state(state) as record:
        if record.read() == "stopped":
            raise SystemExit(125)
        _save(record, str(os.getpid()))
    os.execvp("bash", ["bash", "-lc", command])


def stop(state: str) -> None:
    with _locked_state(state) as record:
        value = record.read()
        # Retain the tombstone until sandbox teardown: a lost request may start late.
        _save(record, "stopped")
        if not value or value == "stopped":
            return
        group = int(value)
        deadline = time.monotonic() + 5
        while _group_alive(group):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("grader process group did not stop")
            time.sleep(0.02)


def _locked_state(state: str):
    record = os.fdopen(os.open(state, os.O_RDWR | os.O_CREAT, 0o600), "r+")
    fcntl.flock(record, fcntl.LOCK_EX)
    return record


def _save(record, value: str) -> None:
    record.seek(0)
    record.truncate()
    record.write(value)
    record.flush()


def _group_alive(group: int) -> bool:
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] not in {"Z", "X"}:
            return True
    return False


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        stop(sys.argv[2])
