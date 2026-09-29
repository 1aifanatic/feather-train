import hashlib
import json
import re
import shlex

import pytest
from feather_harness import FileTools
from feather_harness.tools.process import ProcessTools

from feather_train import E2BRuntime


@pytest.mark.parametrize("content", ["a" * 10_000_000, "é" * 1_000_000], ids=["10MB", "unicode"])
async def test_large_file_arguments_arrive_intact(sandbox, content):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        files = FileTools(transport)
        result = await files.write_tool().invoke(
            json.dumps({"file_path": "large.txt", "content": content})
        )
        assert result.error is None
    result = await sandbox.commands.run(
        f"docker exec {sandbox.container} sha256sum /filesystem/large.txt"
    )
    assert result.stdout.split()[0] == hashlib.sha256(content.encode()).hexdigest()


@pytest.mark.parametrize("padding", [0, 40_000], ids=["inline", "uploaded"])
async def test_bash_preserves_stdin_and_shell_arguments(sandbox, padding):
    command = "#" + "padding" * padding + "\n"
    command += """printf '%s:%s\\n' "$0" "$#"
if read -r input; then exit 9; fi
cat <<'TEXT'
quotes " ' $literal $(printf wrong) `printf wrong` café
TEXT
"""
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        uploads = len(sandbox.files.written_paths)
        result = await ProcessTools(transport).bash_tool().invoke(json.dumps({"command": command}))
        assert len(sandbox.files.written_paths) - uploads == (1 if padding else 0)
    assert "Exit code: 0" in result.content
    assert result.content.endswith(
        "Output:\nbash:0\nquotes \" ' $literal $(printf wrong) `printf wrong` café\n"
    )


@pytest.mark.parametrize("extra_byte", [0, 1], ids=["at-limit", "over-limit"])
async def test_inline_limit_counts_complete_quoted_command(sandbox, extra_byte):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        bash = ProcessTools(transport).bash_tool()
        await bash.invoke(json.dumps({"command": "#"}))
        baseline = next(cmd for cmd in sandbox.commands.started if "remote.py run " in cmd)
        padding = 32 * 1024 - len(baseline.encode()) + extra_byte
        # Apostrophes expand under shell quoting; count the actual launch bytes.
        quotes, rest = divmod(padding, 5)
        uploads = len(sandbox.files.written_paths)
        result = await bash.invoke(json.dumps({"command": "#" + "'" * quotes + "x" * rest}))
        assert "Exit code: 0" in result.content
        assert len(sandbox.files.written_paths) - uploads == extra_byte
        launch = [cmd for cmd in sandbox.commands.started if "remote.py run " in cmd][-1]
        assert len(launch.encode()) <= 32 * 1024


async def test_large_bash_preserves_trailing_newlines_in_heredoc(sandbox):
    command = "#" + "padding" * 40_000 + "\ncat <<EOF\na\n\n"
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        result = await ProcessTools(transport).bash_tool().invoke(json.dumps({"command": command}))
    assert "Exit code: 0" in result.content
    assert result.content.endswith("a\n\n")


async def test_large_bash_rejects_nul_without_executing(sandbox):
    command = "touch /filesystem/should-not-exist\n#" + "x" * 140_000 + "\0"
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        result = await ProcessTools(transport).bash_tool().invoke(json.dumps({"command": command}))
        assert result.error == "tool_error"
        assert result.content == "The command must not contain NUL bytes."
    await sandbox.commands.run(
        f"docker exec {sandbox.container} test ! -e /filesystem/should-not-exist"
    )


async def test_large_argument_error_spills_inside_agent_container(sandbox):
    text = "é" * 40_000
    expected = (
        f"Invalid `new_string` parameter: {text}. old_string and new_string must be different."
    )
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        tool = FileTools(transport).edit_tool()
        result = await tool.invoke(
            json.dumps({"file_path": "notes.txt", "old_string": text, "new_string": text})
        )
        assert result.error == "tool_error"
        assert len(result.content) == 24_000
        saved = re.search(r"Complete returned output saved at (.+?)\.\]", result.content)[1]
        output = await sandbox.commands.run(
            f"docker exec {sandbox.container} sha256sum {shlex.quote(saved)}"
        )
        assert output.stdout.split()[0] == hashlib.sha256(expected.encode()).hexdigest()
