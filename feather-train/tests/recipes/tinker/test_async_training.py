import asyncio
from itertools import count
from types import SimpleNamespace

import braintrust
import chz
import pytest
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.utils import ml_log
from tokenizers import Tokenizer, models

from feather_train.recipes.tinker.dataset import FeatherDataset
from feather_train.recipes.tinker.train import TrainConfig
from feather_train.tasks.taskset import FeatherTask

from ...clients.helpers import _FakeSamplingClient


class RemoteResult:
    def __init__(self, **values):
        self.value = SimpleNamespace(**values)

    async def result_async(self):
        return self.value


class SamplingClient(_FakeSamplingClient):
    def __init__(self, response_ids):
        super().__init__()
        self.response_ids = response_ids

    async def sample_async(self, **kwargs):
        response = await super().sample_async(**kwargs)
        response.sequences[0].tokens[-1] = next(self.response_ids)
        return response


class TrainingClient:
    def __init__(self, *, fail_optimizer=False):
        self.fail_optimizer = fail_optimizer
        self.minibatches = []
        self.updates = 0
        self.samplers = []
        self.response_ids = count(100)

    async def save_state_async(self, name, **kwargs):
        return RemoteResult(path=f"tinker://test/state/{name}")

    async def save_weights_for_sampler_async(self, name, **kwargs):
        return RemoteResult(path=f"tinker://test/sampler/{name}")

    def create_sampling_client(self, path):
        sampler = SamplingClient(self.response_ids)
        self.samplers.append(sampler)
        return sampler

    async def save_weights_and_get_sampling_client_async(self):
        return self.create_sampling_client(f"updated-{self.updates}")

    async def forward_backward_async(self, data, **kwargs):
        self.minibatches.append(data)
        return RemoteResult(loss_fn_outputs=[
            {"logprobs": datum.loss_fn_inputs["logprobs"]} for datum in data
        ])

    async def optim_step_async(self, params):
        if self.fail_optimizer:
            raise RuntimeError("remote optimizer unavailable")
        self.updates += 1
        return RemoteResult(metrics={})


class TrainingLogger(ml_log.Logger):
    def __init__(self):
        self.steps = []
        self.first_update = asyncio.Event()

    def log_hparams(self, config):
        pass

    def log_metrics(self, metrics, step=None):
        self.steps.append(step)
        self.first_update.set()

@pytest.mark.parametrize(
    ("start_batch", "logging_fails"),
    [(0, False), (5, False), (0, True)],
)
async def test_async_training_overlaps_rollouts_and_tags_consumed_batches(
    setup_rollouts, tmp_path, start_batch, logging_fails, monkeypatch,
):
    updates = []
    def update_span(exported, **event):
        updates.append((exported, event))
        if logging_fails:
            raise OSError("Braintrust unavailable")

    monkeypatch.setattr(braintrust, "update_span", update_span)
    logger = TrainingLogger()
    initial_groups_started = asyncio.Event()
    initial_groups = set()

    async def stagger_groups(trace):
        trace.info["bt_span_export"] = trace.id
        idx = trace.task.data.idx
        if idx < 3:
            initial_groups.add(idx)
            if len(initial_groups) == 3:
                initial_groups_started.set()
            await initial_groups_started.wait()
        if idx in {2, 3, 4}:
            await logger.first_update.wait()

    runner, builder, harness = setup_rollouts(concurrency=6, before_sample=stagger_groups)
    client = TrainingClient()
    dataset, config = training_inputs(builder, tmp_path, start_batch)
    original_step = train.do_train_step_streaming_and_get_sampling_client
    async with runner.installed():
        await run_training(
            dataset, config, client,
            start_batch=start_batch, end_batch=start_batch + 3, logger=logger,
        )

    trained = {
        exported: event["metadata"]["training_batch_id"]
        for exported, event in updates if "metadata" in event
    }
    # Unique completion tokens identify the actual minibatch each rollout trained in.
    batches_by_response = {
        datum.loss_fn_inputs["target_tokens"].to_torch().tolist()[-1]: batch
        for batch, minibatch in enumerate(client.minibatches, start=start_batch)
        for datum in minibatch
    }
    assert trained == {
        trace.id: batches_by_response[trace.nodes[-1].token_ids[-1]]
        for trace in harness.traces
    }
    assert train.do_train_step_streaming_and_get_sampling_client is original_step
    assert initial_groups == {0, 1, 2}
    assert harness.peak == 6
    assert client.updates == 3
    assert all(trace.reward == pytest.approx(0.98) for trace in harness.traces)
    assert [len(batch) for batch in client.minibatches] == [4, 4, 4]
    assert logger.steps == list(range(start_batch, start_batch + 3))
    versions = {}
    for trace in harness.traces:
        versions.setdefault(trace.task.data.idx, set()).add(trace.task.data.sampled_policy_version)
    assert all(versions[idx] == {start_batch} for idx in range(3))
    assert len(versions[5]) == 1
    assert next(iter(versions[5])) > start_batch
    assert all(len(versions_for_group) == 1 for versions_for_group in versions.values())
    assert len(client.samplers) == 4
    assert sum(len(s.requests) for s in client.samplers) == 12


async def test_optimizer_failure_drains_cookbook_workers_and_partial_episodes(
    setup_rollouts, tmp_path, monkeypatch,
):
    updates = []
    monkeypatch.setattr(braintrust, "update_span", lambda exported, **event: updates.append(event))
    logger = TrainingLogger()
    blocked = asyncio.Event()

    async def block_later_groups(trace):
        trace.info["bt_span_export"] = trace.id
        if trace.task.data.idx >= 2:
            blocked.set()
            await asyncio.Event().wait()
        else:
            await blocked.wait()

    runner, builder, harness = setup_rollouts(concurrency=6, before_sample=block_later_groups)
    client = TrainingClient(fail_optimizer=True)
    dataset, config = training_inputs(builder, tmp_path)
    tasks_before = asyncio.all_tasks()
    original_async_training = train.do_async_training
    with pytest.raises(RuntimeError, match="remote optimizer unavailable"):
        async with runner.installed():
            await run_training(
                dataset, config, client,
                start_batch=5, end_batch=8, logger=logger,
            )
    assert train.do_async_training is original_async_training
    assert all("metadata" not in event for event in updates)
    assert harness.active == 0
    assert not (asyncio.all_tasks() - tasks_before)
    assert all(harness.braintrust.active_rollout(t.id) is None for t in harness.traces)


def training_inputs(builder, tmp_path, start_batch=5):
    # Prefix batches have already been consumed by the resumed run.
    indices = [99] * (start_batch * 2) + list(range(6))
    tasks = [FeatherTask(builder.task.data.model_copy(update={"idx": idx})) for idx in indices]
    dataset = FeatherDataset(tasks, builder.task.config.length_penalty, groups_per_batch=2)
    recipe = TrainConfig(
        model="test-model",
        renderer_name="qwen3",
        taskset={"hf_dataset": "test/tasks", "hf_revision": "a" * 40},
        length_penalty=builder.task.config.length_penalty,
        log_path=tmp_path,
        max_tokens=7,
        max_steps=start_batch + 3,
        group_size=2,
        groups_per_batch=2,
        max_inflight_rollout_groups=3,
        max_steps_off_policy=2,
    )
    config = chz.replace(
        recipe.cookbook(dataset), num_groups_to_log=0, rollout_json_export=False,
    )
    return dataset, config


def tokenizer():
    return Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))


async def run_training(dataset, config, client, *, start_batch, end_batch, logger):
    await asyncio.wait_for(
        train.do_async_training(
            start_batch=start_batch,
            end_batch=end_batch,
            num_batches=len(dataset),
            config=config,
            training_client=client,
            kl_reference_client=None,
            evaluators=[],
            dataset=dataset,
            ml_logger=logger,
            tokenizer=tokenizer(),
            checkpoint_mgr=checkpoint_utils.CheckpointManager(client, None, config.log_path),
        ),
        timeout=10,
    )
