"""Train Feather with Tinker: python -m feather_train.recipes.tinker.train config.json."""

from __future__ import annotations

import argparse
import asyncio
import logging
from copy import deepcopy
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, ValidationInfo, field_validator
from renderers import RendererPool, config_from_name, create_renderer
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train as cookbook
from tinker_cookbook.tokenizer_utils import get_tokenizer

from feather_train.environment import FeatherEnv
from feather_train.grading.length_penalty import LengthPenaltyConfig
from feather_train.harness import FeatherHarnessConfig
from feather_train.tasks.taskset import FeatherTasksetConfig

from .config import TrainingConfig
from .dataset import FeatherDataset
from .rollout import FeatherRollouts


class TrainConfig(TrainingConfig):
    taskset: FeatherTasksetConfig
    harness: FeatherHarnessConfig = Field(default_factory=FeatherHarnessConfig)
    length_penalty: LengthPenaltyConfig
    behavior_if_exists: Literal["raise", "resume"] = "raise"
    group_size: int = Field(default=8, ge=2)
    max_steps: int = Field(gt=0)
    save_every: int = Field(default=5, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    num_substeps: Literal[1] = 1
    eval_every: Literal[0] = 0
    rollout_max_retries: int = Field(default=2, ge=0)
    loss_fn_config: dict[str, float] | None = Field(default=None, validate_default=True)

    @field_validator("loss_fn_config")
    @classmethod
    def default_cispo_clipping(
        cls, value: dict[str, float] | None, info: ValidationInfo
    ) -> dict[str, float] | None:
        if value is None and info.data.get("loss_fn") == "cispo":
            return {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
        return value

    def environment(self) -> FeatherEnv:
        return FeatherEnv(
            vf.EnvConfig(
                taskset=self.taskset,
                harness=self.harness,
                retries=vf.RetryConfig(
                    rollout=vf.RolloutRetryConfig(
                        max_retries=self.rollout_max_retries,
                        exclude=["NonRetryableHarnessError"],
                    )
                ),
            )
        )



async def train(config: TrainConfig) -> None:
    log_path = config.log_path.expanduser()
    cli_utils.check_log_dir(str(log_path), behavior_if_exists=config.behavior_if_exists)
    renderer_config = config_from_name(config.renderer_name)
    env = config.environment()
    tasks = await asyncio.to_thread(env.taskset.select)
    if not tasks:
        raise ValueError("No training tasks selected")
    dataset = FeatherDataset(
        tasks,
        config.length_penalty,
        config.groups_per_batch,
    )
    if config.max_steps > len(dataset):
        raise ValueError(
            f"{config.max_steps} steps require {config.max_steps * config.groups_per_batch} tasks "
            f"at {config.groups_per_batch} groups per batch; selected {len(tasks)}"
        )
    tokenizer = await asyncio.to_thread(get_tokenizer, config.model)
    renderer = await asyncio.to_thread(
        RendererPool,
        lambda: create_renderer(deepcopy(tokenizer), renderer_config),
        1,
    )
    runner = FeatherRollouts(
        env,
        renderer,
        model=config.model,
        group_size=config.group_size,
        concurrency=config.max_inflight_rollout_groups * config.group_size,
        max_context_length=config.harness.compaction.context_window_tokens,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    log_path.mkdir(parents=True, exist_ok=True)
    with (log_path / "feather-configs.jsonl").open("a") as history:
        history.write(config.model_dump_json() + "\n")
    async with env.serving(), runner.installed():
        await cookbook.main(config.cookbook(dataset))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Feather training configuration (JSON)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = TrainConfig.model_validate_json(args.config.read_text())
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
