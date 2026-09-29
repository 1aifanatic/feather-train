from unittest.mock import patch

import pytest

from feather_train.recipes.tinker.dataset import FeatherDataset
from feather_train.recipes.tinker.train import TrainConfig, train
from feather_train.tasks.taskset import FeatherTaskset


def config(tmp_path, **overrides):
    return TrainConfig(
        model="test-policy",
        renderer_name="qwen3",
        taskset={"hf_dataset": "test/tasks", "hf_revision": "a" * 40},
        harness={"enable_braintrust": False, "compaction": {"context_window_tokens": 65536}},
        length_penalty={"output_token_reference_length": 40000, "num_output_tokens_weight": 0.075},
        log_path=tmp_path,
        learning_rate=1e-5,
        max_steps=2,
        **overrides,
    )


def test_recipe_passes_training_limits_without_enabling_a_second_retry_or_reward_policy(tmp_path):
    recipe = config(tmp_path, rollout_max_retries=3)
    env = recipe.environment()
    assert env.harness_timeout is None
    assert env.config.retries.rollout.max_retries == 3
    assert env.config.retries.rollout.exclude == ["NonRetryableHarnessError"]
    dataset = FeatherDataset([], recipe.length_penalty, 1)
    training = recipe.cookbook(dataset)
    assert training.renderer_name == recipe.renderer_name
    assert training.loss_fn == "cispo"
    assert training.loss_fn_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    assert training.num_substeps == 1
    assert training.rollout_error_tolerance is False
    assert training.kl_penalty_coef == 0
    assert training.async_config.groups_per_batch == recipe.max_inflight_rollout_groups
    assert training.async_config.max_steps_off_policy == recipe.max_steps_off_policy
    assert training.stream_minibatch_config.groups_per_batch == recipe.groups_per_batch
    assert training.stream_minibatch_config.num_minibatches == 1


@pytest.mark.parametrize(
    "loss, options",
    [
        pytest.param("importance_sampling", None, id="alternative-loss"),
        pytest.param("cispo", {"clip_high_threshold": 2.0}, id="explicit-clipping"),
    ],
)
def test_loss_settings_are_not_overwritten_by_default_cispo_clipping(tmp_path, loss, options):
    recipe = config(tmp_path, loss_fn=loss, loss_fn_config=options)
    training = recipe.cookbook(FeatherDataset([], recipe.length_penalty, 1))
    assert training.loss_fn == loss
    assert training.loss_fn_config == options


async def test_incomplete_training_batch_fails_before_model_setup(tmp_path, setup_rollouts):
    _, builder, _ = setup_rollouts()
    recipe = config(tmp_path / "run", groups_per_batch=16)
    with (
        patch.object(FeatherTaskset, "select", return_value=[builder.task] * 17),
        patch("feather_train.recipes.tinker.train.get_tokenizer") as tokenizer,
    ):
        with pytest.raises(ValueError, match="2 steps require 32 tasks.*selected 17"):
            await train(recipe)
    tokenizer.assert_not_called()
