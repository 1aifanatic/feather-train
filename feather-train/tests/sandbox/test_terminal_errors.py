from types import SimpleNamespace

import pytest
from e2b.exceptions import (
    AuthenticationException,
    InvalidArgumentException,
    NotEnoughSpaceException,
    NotFoundException,
    TemplateException,
)
from feather_harness.tools.process import ProcessTools

from feather_train import E2BRuntime


@pytest.mark.parametrize(
    "error_type",
    [
        AuthenticationException,
        InvalidArgumentException,
        NotEnoughSpaceException,
        NotFoundException,
        TemplateException,
    ],
)
async def test_terminal_launch_error_is_preserved_without_result_recovery(error_type):
    failure = error_type("provider rejected the launch")
    launches = 0
    inspections = 0

    class Commands:
        async def run(self, command, **kwargs):
            nonlocal launches, inspections
            if "remote.py run " in command:
                launches += 1
                raise failure
            if "remote.py inspect " in command:
                inspections += 1
                raise RuntimeError("no result exists for the rejected launch")
            return SimpleNamespace(stdout="")

    async with E2BRuntime(SimpleNamespace(commands=Commands())) as runtime:
        with pytest.raises(error_type) as raised:
            await ProcessTools(runtime).bash_tool().invoke('{"command":"echo once"}')

    assert raised.value is failure
    assert launches == 1
    assert inspections == 0
