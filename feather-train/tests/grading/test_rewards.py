"""Trajectory penalties and the score consumed by Verifiers."""

from types import SimpleNamespace

import pytest
from verifiers.v1 import HarnessError

from feather_train.grading.rewards import (
    TrajectoryRewardConfig,
    adjust_trajectory_reward,
    harbor_grader_reward,
)

from .helpers import task


def test_invalid_call_penalty_is_capped_and_can_make_reward_negative():
    result = adjust_trajectory_reward(
        0.25,
        tool_call_count=3,
        schema_invalid_tool_call_count=2,
        unknown_tool_call_count=1,
        config=TrajectoryRewardConfig(
            invalid_tool_call_penalty=0.3, max_invalid_tool_call_penalty=0.5
        ),
    )
    assert result.value == -0.25
    assert result.penalty == 0.5


def test_uncapped_penalty_counts_schema_invalid_and_unknown_calls():
    result = adjust_trajectory_reward(
        0.9,
        tool_call_count=5,
        schema_invalid_tool_call_count=2,
        unknown_tool_call_count=1,
        config=TrajectoryRewardConfig(invalid_tool_call_penalty=0.1),
    )
    assert result.penalty == pytest.approx(0.3)
    assert result.value == pytest.approx(0.6)
    assert not result.zero_tool_calls


@pytest.mark.parametrize(("enabled", "expected"), [(False, 0.75), (True, 0.0)])
def test_zero_tool_policy_is_opt_in(enabled, expected):
    result = adjust_trajectory_reward(
        0.75,
        tool_call_count=0,
        schema_invalid_tool_call_count=0,
        unknown_tool_call_count=0,
        config=TrajectoryRewardConfig(zero_tool_calls=enabled),
    )
    assert result.value == expected
    assert result.penalty == 0
    assert result.zero_tool_calls is enabled


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"grading_score": 0.75, "training_reward": 0.0}, 0.0),
        ({"grading_score": 0.75, "training_reward": 0.5}, 0.5),
        ({"grading_score": 0.75}, 0.75),
    ],
)
async def test_verifiers_reads_shaped_reward_or_falls_back_to_grade(info, expected):
    assert await harbor_grader_reward(task(), SimpleNamespace(info=info)) == expected


@pytest.mark.parametrize(
    ("stop_condition", "message"),
    [
        ("agent_completed", "did not produce grading_score"),
        ("harness_timeout", "timed out before grading"),
    ],
)
async def test_verifiers_rejects_missing_grade_even_with_training_reward(stop_condition, message):
    trace = SimpleNamespace(info={"training_reward": 0.5}, stop_condition=stop_condition)
    with pytest.raises(HarnessError, match=message):
        await harbor_grader_reward(task(), trace)
