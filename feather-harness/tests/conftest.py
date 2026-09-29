import pytest

from feather_harness import LocalShellRuntime, ProcessTools


@pytest.fixture
async def process_tools(tmp_path):
    async with LocalShellRuntime(tmp_path, log_directory=tmp_path / "logs") as transport:
        yield ProcessTools(transport)
