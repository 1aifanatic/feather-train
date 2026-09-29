"""Bad baked content is permanent; a failed inspection can still be transient."""

import pytest
from e2b.exceptions import (
    AuthenticationException,
    InvalidArgumentException,
    NotFoundException,
    TemplateException,
)
from verifiers.v1 import SandboxError

from feather_train.errors import NonRetryableHarnessError
from feather_train.grading.grader import GradingConfig, HarborGradingError, grade_task
from feather_train.sandbox.lifecycle import SandboxIO

from .helpers import GraderSandbox, task


@pytest.mark.parametrize(
    "has_test_script", [False, True], ids=["missing-tests", "missing-dependencies"]
)
async def test_bad_bake_is_not_retryable(sandbox, has_test_script):
    session = SandboxIO(sandbox, container=sandbox.container)
    if has_test_script:
        # This tool image intentionally has none of the grader's pinned dependencies.
        await session.exec(
            'mkdir -p /opt/apex_grader_tests && printf "exit 0\\n" > /opt/apex_grader_tests/test.sh',
            shell=True,
        )
    with pytest.raises(HarborGradingError) as raised:
        await grade_task(
            task(),
            session,
            instruction="Solve this exactly.",
            config=GradingConfig(),
            final_answer="",
        )
    assert raised.value.retryable is False
    assert isinstance(raised.value.__cause__, NonRetryableHarnessError)
    assert not any(step.name == "run_tests" for step in raised.value.steps)


async def test_dependency_check_connection_failure_remains_retryable():
    class DisconnectedSandbox(GraderSandbox):
        async def exec(self, *args, timeout=None):
            if "import importlib.metadata" in args[-1]:
                raise SandboxError("connection lost")
            return await super().exec(*args, timeout=timeout)

    with pytest.raises(HarborGradingError) as raised:
        await grade_task(
            task(),
            DisconnectedSandbox(),
            instruction="Solve this exactly.",
            config=GradingConfig(),
            final_answer="",
        )
    assert raised.value.retryable is True
    assert isinstance(raised.value.__cause__, SandboxError)


@pytest.mark.parametrize(
    "error_type",
    [
        AuthenticationException,
        InvalidArgumentException,
        NotFoundException,
        TemplateException,
    ],
)
async def test_permanent_overlay_upload_failure_is_not_retryable(error_type):
    failure = error_type("upload rejected")

    class RejectedUpload(GraderSandbox):
        async def extract_archive(self, *args, **kwargs):
            raise failure

    sandbox = RejectedUpload()
    with pytest.raises(HarborGradingError) as raised:
        await grade_task(
            task(),
            sandbox,
            instruction="Solve this exactly.",
            config=GradingConfig(),
            final_answer="",
        )
    assert raised.value.retryable is False
    assert raised.value.__cause__ is failure
    assert sandbox.test_runs == 0
