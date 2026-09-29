"""Trajectory reward shaping and the score consumed by Verifiers."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from verifiers.v1 import HarnessError, TaskData, Trace


class TrajectoryRewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invalid_tool_call_penalty: float = Field(
        0.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    max_invalid_tool_call_penalty: float = Field(
        1.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    zero_tool_calls: bool = False


@dataclass(frozen=True, slots=True)
class TrajectoryReward:
    value: float
    penalty: float
    zero_tool_calls: bool


def adjust_trajectory_reward(
    grading_score: float,
    *,
    tool_call_count: int,
    schema_invalid_tool_call_count: int,
    unknown_tool_call_count: int,
    config: TrajectoryRewardConfig,
) -> TrajectoryReward:
    if config.zero_tool_calls and tool_call_count == 0:
        return TrajectoryReward(value=0.0, penalty=0.0, zero_tool_calls=True)

    penalty = min(
        config.invalid_tool_call_penalty
        * (schema_invalid_tool_call_count + unknown_tool_call_count),
        config.max_invalid_tool_call_penalty,
    )
    return TrajectoryReward(
        value=grading_score - penalty,
        penalty=penalty,
        zero_tool_calls=False,
    )


async def harbor_grader_reward(task: TaskData, trace: Trace) -> float:
    """Read the pre-computed final score from trace metadata.

    The harness invokes the harbor grader (test.sh -> grader.py) inside the
    sandbox *before* the reward fn runs (so the sandbox is still alive) and
    stashes the raw score and any configured trajectory adjustment on the trace.
    """
    if trace.info.get("grading_score") is None:
        task_id = getattr(task, "task_id", "<unknown>")
        if trace.stop_condition == "harness_timeout":
            raise HarnessError(f"Feather harness timed out before grading task {task_id}")
        raise HarnessError(f"harbor grader did not produce grading_score for task {task_id}")
    return float(trace.info.get("training_reward", trace.info["grading_score"]))
