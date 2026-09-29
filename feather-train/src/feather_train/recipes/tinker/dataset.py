"""Loaded Feather tasks exposed through Cookbook's dataset interfaces."""

from dataclasses import dataclass

import chz
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder

from feather_train.grading.length_penalty import LengthPenaltyConfig
from feather_train.tasks.taskset import FeatherTask
from feather_train.tasks.training import FeatherTrainingTask, FeatherTrainingTaskConfig


@dataclass
class FeatherGroupBuilder(EnvGroupBuilder):
    task: FeatherTask

    async def make_envs(self) -> list[Env]:
        raise RuntimeError("Feather groups require the recipe's scoped rollout adapter")


class FeatherDataset(RLDataset):
    def __init__(
        self,
        tasks: list[FeatherTask],
        length_penalty: LengthPenaltyConfig | None,
        groups_per_batch: int,
    ) -> None:
        self.groups_per_batch = groups_per_batch
        self.builders = [
            FeatherGroupBuilder(
                FeatherTrainingTask(
                    task.data,
                    FeatherTrainingTaskConfig(
                        judges=task.config.judges, length_penalty=length_penalty
                    ),
                )
                if length_penalty is not None
                else task
            )
            for task in tasks
        ]

    def to_dict(self) -> dict[str, int]:
        # Cookbook logs datasets as configuration; task contents belong in rollout traces.
        return {"num_tasks": len(self.builders), "groups_per_batch": self.groups_per_batch}

    def get_batch(self, index: int) -> list[FeatherGroupBuilder]:
        start = index * self.groups_per_batch
        return self.builders[start : start + self.groups_per_batch]

    def __len__(self) -> int:
        return len(self.builders) // self.groups_per_batch


@chz.chz
class FeatherDatasetBuilder(RLDatasetBuilder):
    dataset: FeatherDataset

    async def __call__(self) -> tuple[FeatherDataset, None]:
        return self.dataset, None
