import json

import pytest
from feather_harness import FileTools

from feather_train import E2BRuntime


@pytest.mark.parametrize("run_as", [None, "65534"])
async def test_file_tools_use_shared_semantics_in_container(sandbox, run_as):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime, run_as=run_as
    ) as transport:
        files = FileTools(transport)
        await files.write_tool().invoke(
            json.dumps({"file_path": "notes.txt", "content": "hello\r\nworld\r\n"})
        )
        result = await files.read_tool().invoke(json.dumps({"file_path": "/filesystem/notes.txt"}))
        assert result.content == "Contents of notes.txt:\n     1\thello\n     2\tworld\n"
        await files.edit_tool().invoke(
            json.dumps(
                {"file_path": "./notes.txt", "old_string": "hello\nworld", "new_string": "héllo"}
            )
        )
        result = await files.grep_tool().invoke(
            json.dumps({"pattern": "héllo", "include": "*.txt"})
        )
        assert result.content == "notes.txt:1:héllo"
        assert (
            await files.glob_tool().invoke(json.dumps({"pattern": "*.txt"}))
        ).content == "notes.txt"
        assert (
            await files.read_tool().invoke(json.dumps({"file_path": "/etc/passwd"}))
        ).error == "invalid_path"
        await files.write_tool().invoke(json.dumps({"file_path": "notes.txt", "content": "done"}))
    result = await sandbox.commands.run(
        f"docker exec {sandbox.container} cat /filesystem/notes.txt"
    )
    assert result.stdout == "done"


async def test_external_grants_are_normalized_like_local_transport(sandbox):
    await sandbox.commands.run(f"docker exec {sandbox.container} mkdir -p /shared")
    async with E2BRuntime(
        sandbox,
        container=sandbox.container,
        host_runtime=sandbox.host_runtime,
        write_grants=("/unused/../shared",),
    ) as transport:
        files = FileTools(transport)
        await files.write_tool().invoke(
            json.dumps({"file_path": "/shared/answer.txt", "content": "answer"})
        )
        result = await files.read_tool().invoke(json.dumps({"file_path": "/shared/answer.txt"}))
        assert result.content == "Contents of /shared/answer.txt:\n     1\tanswer\n"
