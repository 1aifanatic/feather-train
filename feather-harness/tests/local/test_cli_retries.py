import json

import pytest

from tests.local.test_cli import cli, model_server, response


@pytest.mark.parametrize("profile", ["train", "prod"])
def test_inference_retries_resend_the_request_without_replaying_a_mutation(tmp_path, profile):
    replies = [
        (200, response(tools=[("bash", {"command": "printf 'once\\n' >> effects.txt"})])),
        (504, {"error": {"message": "temporary gateway failure"}}),
        (429, {"error": {"message": "temporary rate limit"}}),
        (200, response("Done.")),
    ]
    with model_server(replies) as (url, requests):
        completed = cli(
            tmp_path, "--model", "test", "--base-url", url, "--profile", profile, "Do the task"
        )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "effects.txt").read_text() == "once\n"
    assert json.loads(completed.stdout)["num_turns"] == 2
    assert len(requests) == 4
    assert requests[1] == requests[2] == requests[3]


def test_non_retryable_authentication_failure_is_not_retried(tmp_path):
    with model_server([(401, {"error": {"message": "invalid credentials"}})]) as (url, requests):
        completed = cli(tmp_path, "--model", "test", "--base-url", url, "Do the task")

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["stop_reason"] == "failed"
    assert len(requests) == 1
