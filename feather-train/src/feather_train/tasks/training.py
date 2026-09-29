"""Opt-in group reward shaping for training with Verifiers 0.2.1."""

import verifiers.v1 as vf

from feather_train.grading.length_penalty import LengthPenaltyConfig, calculate_length_penalties
from feather_train.tasks.taskset import FeatherTask, FeatherTaskData


class FeatherTrainingTaskConfig(vf.TaskConfig):
    length_penalty: LengthPenaltyConfig


class FeatherTrainingTask(
    FeatherTask, vf.Task[FeatherTaskData, vf.State, FeatherTrainingTaskConfig]
):
    """Training-only scoring; eval.py uses FeatherTask without group shaping."""

    @vf.group_reward
    async def length_penalty(self, traces: list[vf.Trace]) -> list[float]:
        penalties = calculate_length_penalties(traces, self.config.length_penalty)
        return [-penalty for penalty in penalties]
