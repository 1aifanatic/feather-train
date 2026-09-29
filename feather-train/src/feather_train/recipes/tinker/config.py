"""Validated training settings and their single conversion to Cookbook."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tinker_cookbook.rl import train as cookbook

from feather_train.grading.length_penalty import LengthPenaltyConfig

from .dataset import FeatherDataset, FeatherDatasetBuilder


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    model: str
    renderer_name: str
    log_path: Path
    behavior_if_exists: Literal["raise", "resume", "delete"] = "raise"
    length_penalty: LengthPenaltyConfig | None = None
    group_size: int = Field(default=8, ge=1)
    groups_per_batch: int = Field(default=16, ge=1)
    max_inflight_rollout_groups: int = Field(default=32, ge=1)
    max_steps_off_policy: int = Field(default=4, ge=0)
    max_tokens: int = Field(default=131072, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    learning_rate: float = Field(default=1e-5, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    loss_fn: Literal["importance_sampling", "ppo", "cispo", "dro"] = "cispo"
    loss_fn_config: dict[str, float] | None = None
    lora_rank: int = Field(default=64, gt=0)
    num_substeps: int = Field(default=1, ge=1)
    save_every: int = Field(default=5, ge=0)
    eval_every: int = Field(default=0, ge=0)
    load_checkpoint_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    ttl_seconds: int | None = Field(default=604800, gt=0)
    num_groups_to_log: int = Field(default=4, ge=0)
    compute_post_kl: bool = False
    enable_trace: bool = False
    remove_constant_reward_groups: bool = False
    rollout_json_export: bool = True
    rollout_error_tolerance: Literal[False] = False
    # Reference-policy sampling is not implemented by this recipe.
    kl_penalty_coef: Literal[0.0] = 0.0
    kl_discount_factor: float | None = 0.0

    @model_validator(mode="after")
    def validate_training_settings(self) -> Self:
        if self.length_penalty is not None and self.group_size < 2:
            raise ValueError("Group length penalties require group_size >= 2")
        if self.groups_per_batch % self.num_substeps:
            raise ValueError("groups_per_batch must be divisible by num_substeps")
        return self

    def cookbook(self, dataset: FeatherDataset) -> cookbook.Config:
        return cookbook.Config(
            learning_rate=self.learning_rate,
            dataset_builder=FeatherDatasetBuilder(dataset=dataset),
            model_name=self.model,
            recipe_name="feather_tinker",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            log_path=str(self.log_path.expanduser()),
            max_steps=self.max_steps,
            loss_fn=self.loss_fn,
            loss_fn_config=self.loss_fn_config,
            lora_rank=self.lora_rank,
            num_substeps=self.num_substeps,
            save_every=self.save_every,
            eval_every=self.eval_every,
            evaluator_builders=[],
            load_checkpoint_path=self.load_checkpoint_path,
            wandb_project=self.wandb_project,
            wandb_name=self.wandb_name,
            renderer_name=self.renderer_name,
            base_url=self.base_url,
            ttl_seconds=self.ttl_seconds,
            num_groups_to_log=self.num_groups_to_log,
            compute_post_kl=self.compute_post_kl,
            enable_trace=self.enable_trace,
            remove_constant_reward_groups=self.remove_constant_reward_groups,
            rollout_json_export=self.rollout_json_export,
            rollout_error_tolerance=self.rollout_error_tolerance,
            kl_penalty_coef=self.kl_penalty_coef,
            kl_discount_factor=self.kl_discount_factor,
            # Streaming uses AsyncConfig's batch size as the group worker count.
            async_config=cookbook.AsyncConfig(
                max_steps_off_policy=self.max_steps_off_policy,
                groups_per_batch=self.max_inflight_rollout_groups,
            ),
            stream_minibatch_config=cookbook.StreamMinibatchConfig(
                groups_per_batch=self.groups_per_batch,
                num_minibatches=1,
            ),
        )
