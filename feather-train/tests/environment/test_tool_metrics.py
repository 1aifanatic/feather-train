from feather_harness import FileTools, LocalFileRuntime

from feather_train.agent import ToolMetrics


async def test_file_errors_do_not_count_as_invalid_arguments(tmp_path):
    (tmp_path / "notes.txt").write_text("one line")
    files = FileTools(LocalFileRuntime(tmp_path))
    metrics = ToolMetrics()
    read = metrics.wrap(files.read_tool())

    past_eof = await read.invoke('{"file_path": "notes.txt", "offset": 2}')
    assert past_eof.error == "tool_error"
    assert metrics.invalid_calls == 0

    invalid_offset = await read.invoke('{"file_path": "notes.txt", "offset": 0}')
    assert invalid_offset.error == "invalid_arguments"
    assert metrics.invalid_calls == 1
    assert metrics.known_calls == 2
