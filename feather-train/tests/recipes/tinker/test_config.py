"""Rules shared by public JSON, private Tinker and Loops launchers."""

import pytest
from pydantic import ValidationError

from feather_train.recipes.tinker.config import TrainingConfig
from feather_train.recipes.tinker.dataset import FeatherDataset


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"groups_per_batch": 0}, "groups_per_batch"),
        ({"max_inflight_rollout_groups": 0}, "max_inflight_rollout_groups"),
        ({"max_steps_off_policy": -1}, "max_steps_off_policy"),
        ({"rollout_error_tolerance": True}, "rollout_error_tolerance"),
        ({"kl_penalty_coef": 0.1}, "kl_penalty_coef"),
        ({"groups_per_batch": 3, "num_substeps": 2}, "must be divisible by num_substeps"),
        (
            {
                "group_size": 1,
                "length_penalty": {
                    "output_token_reference_length": 40000,
                    "num_output_tokens_weight": 0.075,
                },
            },
            "group_size >= 2",
        ),
    ],
)
def test_invalid_training_settings_fail_at_construction(tmp_path, settings, message):
    with pytest.raises(ValidationError, match=message):
        TrainingConfig(model="policy", renderer_name="qwen3_6", log_path=tmp_path, **settings)


def test_cookbook_conversion_preserves_optimizer_checkpoint_and_logging_settings(tmp_path):
    config = TrainingConfig(
        model="policy",
        renderer_name="qwen3_6",
        log_path=tmp_path,
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.1},
        learning_rate=0.0002,
        lora_rank=8,
        num_substeps=2,
        save_every=0,
        max_steps=None,
        load_checkpoint_path="tinker://run/state/4",
        remove_constant_reward_groups=True,
        compute_post_kl=True,
        enable_trace=True,
        rollout_json_export=False,
        num_groups_to_log=2,
        ttl_seconds=None,
        base_url="https://training.invalid",
        wandb_project="project",
        wandb_name="run",
    )
    cookbook = config.cookbook(FeatherDataset([], None, config.groups_per_batch))
    for name in (
        "renderer_name",
        "loss_fn",
        "loss_fn_config",
        "learning_rate",
        "lora_rank",
        "num_substeps",
        "save_every",
        "max_steps",
        "load_checkpoint_path",
        "remove_constant_reward_groups",
        "compute_post_kl",
        "enable_trace",
        "rollout_json_export",
        "num_groups_to_log",
        "ttl_seconds",
        "base_url",
        "wandb_project",
        "wandb_name",
    ):
        assert getattr(cookbook, name) == getattr(config, name)
