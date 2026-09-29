"""Real Docker tools, with local pipes supplying the E2B SDK command events."""

import asyncio
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from e2b.envd.process import process_pb as pb
from e2b.sandbox_async.commands.command_handle import AsyncCommandHandle
from protobuf import Oneof

from feather_train.sandbox.runtime import build_runtime, install_runtime


def pytest_collection_modifyitems(items):
    for item in items:
        if "sandbox" in item.fixturenames:
            item.add_marker(pytest.mark.docker)


class DockerCommands:
    def __init__(self):
        self.collectors = []
        self.drop_next_reply = False
        self.drop_next_query = False
        self.drop_before_reply = False
        self.started = []

    async def run(
        self, command, *, background=False, stdin=False, on_stdout=None, on_stderr=None, **kwargs
    ):
        self.started.append(command)
        drop_reply = self.drop_next_reply and "remote.py run " in command
        if drop_reply:
            self.drop_next_reply = False
        if self.drop_next_query and "remote.py tasks " in command:
            drop_reply = True
            self.drop_next_query = False
        drop_early = self.drop_before_reply and "remote.py run " in command
        if drop_early:
            self.drop_before_reply = False
        process = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        queue = asyncio.Queue()

        async def copy(stream, name):
            # Deliberately split JSON/UTF-8 across events, as the real network can.
            while chunk := await stream.read(37):
                await queue.put(
                    pb.ProcessEvent(
                        event=Oneof("data", pb.ProcessEvent.DataEvent(output=Oneof(name, chunk)))
                    )
                )

        async def collect():
            await asyncio.gather(copy(process.stdout, "stdout"), copy(process.stderr, "stderr"))
            code = await process.wait()
            await queue.put(
                pb.ProcessEvent(
                    event=Oneof("end", pb.ProcessEvent.EndEvent(exit_code=code, exited=True))
                )
            )

        self.collectors.append(asyncio.create_task(collect()))

        async def events():
            if drop_early:
                raise ConnectionError("connection lost while tool is still running")
            while True:
                event = await queue.get()
                if drop_reply and event.event.field == "data":
                    raise ConnectionError("tool executed but reply connection lost")
                yield pb.StartResponse(event=event)
                if event.event.field == "end":
                    break

        async def send(data, request_timeout=None):
            process.stdin.write(data.encode() if isinstance(data, str) else data)
            await process.stdin.drain()

        async def close_stdin(request_timeout=None):
            process.stdin.close()

        async def kill():
            process.kill()
            return True

        handle = AsyncCommandHandle(
            process.pid,
            kill,
            events(),
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            handle_send_stdin=send,
            handle_close_stdin=close_stdin,
        )
        return handle if background else await handle.wait()


class HostFiles:
    def __init__(self):
        self.written_paths = []

    async def write(self, path, data, **kwargs):
        path = Path(path)
        path.write_bytes(data.read() if hasattr(data, "read") else data)
        self.written_paths.append(path)

    async def remove(self, path, **kwargs):
        Path(path).unlink()


@pytest.fixture
async def sandbox(tmp_path, request):
    name = "feather-test-" + uuid.uuid4().hex[:12]
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--init",
            "--network",
            "none",
            "--user",
            getattr(request, "param", "root"),
            "--name",
            name,
            "feather-e2b-test",
        ],
        check=True,
        capture_output=True,
    )
    sandbox = SimpleNamespace(
        commands=DockerCommands(),
        files=HostFiles(),
        container=name,
        host_runtime=str(tmp_path / "host"),
    )
    try:
        archive = tmp_path / "tools.tar.gz"
        build_runtime(archive)
        await install_runtime(sandbox, archive, container=name, host_runtime=sandbox.host_runtime)
        yield sandbox
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)
        await asyncio.gather(*sandbox.commands.collectors, return_exceptions=True)
