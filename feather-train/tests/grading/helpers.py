"""Controlled judge execution and task input shared by grading tests."""

import base64
import io
import tarfile
from types import SimpleNamespace

from feather_train.sandbox.lifecycle import CommandOutput


class GraderSandbox:
    def __init__(
        self,
        attempts=(0,),
        reward=b"0.75",
        detail=b'{"num_met":3,"num_criteria":4}',
        stdout="grader output",
        stderr="judge unavailable",
    ):
        self.attempts = iter(attempts)
        self.reward = reward
        self.detail = detail
        self.stdout = stdout
        self.stderr = stderr
        self.test_runs = 0
        self.overlays = {}
        self.test_command = ""

    async def exec_grader(self, command, *, timeout):
        return await self.exec("bash", "-lc", command, timeout=timeout)

    async def exec(self, *args, timeout=None):
        command = args[-1]
        if command.endswith("/bin/bash /tests/test.sh"):
            self.test_runs += 1
            self.test_command = command
            outcome = next(self.attempts)
            if isinstance(outcome, BaseException):
                raise outcome
            return CommandOutput(outcome, self.stdout, self.stderr if outcome else "")
        if command.startswith("base64 -w0 "):
            data = self.detail if command.endswith(".json") else self.reward
            return CommandOutput(0, base64.b64encode(data).decode("ascii"), "")
        return CommandOutput(0, "", "")

    async def extract_archive(self, data, destination, *, timeout):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            self.overlays = {entry.name: archive.extractfile(entry).read() for entry in archive}
        return CommandOutput(0, "", "")


def task():
    return SimpleNamespace(
        task_id="task-1",
        world_id="world-1",
        hf_dataset="test/tasks",
        hf_revision="revision",
        rubric=[],
        rubric_doc={"sections": []},
        expected_deliverable={},
        task_artifacts=SimpleNamespace(tests=False),
        source="factory",
        submission_transport="text_response",
        grading_evidence_mode=None,
    )
