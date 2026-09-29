import asyncio
from unittest.mock import patch

import pytest
import verifiers.v1 as vf
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import rollouts, train
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

from feather_train.errors import NonRetryableHarnessError

from ...clients.helpers import _FakeSamplingClient


async def sampler_for_batch(sampler, batch, tmp_path):
    class TrainingClient:
        async def save_weights_and_get_sampling_client_async(self):
            return sampler

    training_client = TrainingClient()
    manager = checkpoint_utils.CheckpointManager(training_client, None, str(tmp_path))
    client, _ = await train.save_checkpoint_and_get_sampling_client(training_client, manager, batch)
    return client


async def run_group(sampler, builder):
    return await train.do_group_rollout_and_filter_constant_reward(
        sampler,
        builder,
        max_tokens=7,
        temperature=0.6,
        do_remove_constant_reward_groups=False,
        enable_logging=False,
    )


async def test_overlapping_policies_keep_their_own_tokens_and_versions(setup_rollouts, tmp_path):
    release = asyncio.Event()
    both_policies_started = asyncio.Event()
    active_versions = set()

    async def wait_for_release(trace):
        active_versions.add(trace.task.data.sampled_policy_version)
        if active_versions == {3, 4}:
            both_policies_started.set()
        await release.wait()

    runner, builder, harness = setup_rollouts(concurrency=4, before_sample=wait_for_release)
    samplers = [_FakeSamplingClient([40, 41]), _FakeSamplingClient([50, 51, 52, 53])]
    async with runner.installed():
        clients = [await sampler_for_batch(s, i + 3, tmp_path) for i, s in enumerate(samplers)]
        tasks = [asyncio.create_task(run_group(s, builder)) for s in clients]
        await asyncio.wait_for(both_policies_started.wait(), 5)
        release.set()
        groups = await asyncio.wait_for(asyncio.gather(*tasks), 5)

    assert builder.task.data.sampled_policy_version is None
    for version, sampler, group in zip([3, 4], samplers, groups, strict=True):
        expected_reward = 1 - len(sampler.completion_ids) * 0.01
        assert group.final_rewards_G == pytest.approx([expected_reward] * 2)
        traces = [t for t in harness.traces if t.task.data.sampled_policy_version == version]
        assert len(traces) == 2
        assert len({t.task.data.rollout_group_id for t in traces}) == 1
        for trajectory in group.trajectories_G:
            (transition,) = trajectory.transitions
            assert transition.ac.tokens == sampler.completion_ids
        for request in sampler.requests:
            params = request["sampling_params"]
            assert (params.max_tokens, params.temperature, params.top_p, params.top_k) == (
                7, 0.6, 0.8, 12,
            )
    assert len({t.task.data.rollout_group_id for t in harness.traces}) == 2


@pytest.mark.parametrize("logging_fails", [False, True], ids=["published", "unavailable"])
async def test_final_group_rewards_are_published_without_making_logging_required(
    setup_rollouts, tmp_path, logging_fails,
):
    async def export_span(trace):
        trace.info["bt_span_export"] = trace.id

    runner, builder, harness = setup_rollouts(before_sample=export_span)
    # Only the external logging call is replaced; episode execution and scoring are real.
    with patch("braintrust.update_span", side_effect=OSError("offline") if logging_fails else None) as publish:
        async with runner.installed():
            sampler = await sampler_for_batch(_FakeSamplingClient(), 0, tmp_path)
            group = await run_group(sampler, builder)
    assert group.final_rewards_G == pytest.approx([0.98, 0.98])
    published = {call.args[0]: call.kwargs["metrics"] for call in publish.call_args_list}
    for trace in harness.traces:
        assert published[trace.id]["reward"] == pytest.approx(0.98)
        assert published[trace.id]["reward_after_length_penalty"] == trace.reward


async def test_verifiers_retries_transient_attempt_before_group_scoring(setup_rollouts, tmp_path):
    attempts = 0

    async def fail_once(trace):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise vf.HarnessError("temporary sandbox outage")

    runner, builder, harness = setup_rollouts(before_sample=fail_once)
    async with runner.installed():
        sampler = await sampler_for_batch(_FakeSamplingClient(), 0, tmp_path)
        group = await run_group(sampler, builder)
    assert attempts == 3
    assert group.final_rewards_G == pytest.approx([0.98, 0.98])
    assert harness.traces[0].error.type == "HarnessError"
    assert all(not t.errors for t in harness.traces[1:])
    assert all(harness.braintrust.active_rollout(t.id) is None for t in harness.traces)


@pytest.mark.parametrize("fatal", [False, True])
async def test_all_failed_rollouts_stop_training_and_fatal_errors_are_not_retried(
    setup_rollouts,
    tmp_path,
    fatal,
):
    async def fail(trace):
        error = NonRetryableHarnessError if fatal else vf.HarnessError
        raise error("sandbox contract failed")

    runner, builder, harness = setup_rollouts(retries=2, before_sample=fail)
    original_rollout = rollouts.do_group_rollout
    original_sampler = train.save_checkpoint_and_get_sampling_client
    with pytest.raises(NonRetryableHarnessError if fatal else ValueError):
        async with runner.installed():
            sampler = await sampler_for_batch(_FakeSamplingClient(), 0, tmp_path)
            await run_group(sampler, builder)
    assert len(harness.traces) == (2 if fatal else 6)
    assert rollouts.do_group_rollout is original_rollout
    assert train.save_checkpoint_and_get_sampling_client is original_sampler
    assert all(harness.braintrust.active_rollout(t.id) is None for t in harness.traces)


async def test_partial_group_rewards_reach_cookbook_training_data(setup_rollouts, tmp_path, monkeypatch):
    monkeypatch.setattr("braintrust.update_span", lambda *args, **kwargs: None)
    attempts = 0

    async def fail_one_rollout(trace):
        trace.info["bt_span_export"] = trace.id
        nonlocal attempts
        attempts += 1
        if attempts > 2:
            raise vf.HarnessError("sandbox unavailable")
        trace.info["grading_score"] = 1.0 if attempts == 1 else 0.0

    class Sampler(_FakeSamplingClient):
        async def sample_async(self, **kwargs):
            self.completion_ids = [40, 41] if not self.requests else [50, 51, 52, 53]
            self.logprobs = [-0.25] * len(self.completion_ids)
            return await super().sample_async(**kwargs)

    runner, builder, harness = setup_rollouts(group_size=3, concurrency=1, before_sample=fail_one_rollout)
    async with runner.installed():
        sampler = await sampler_for_batch(Sampler(), 0, tmp_path)
        group = await run_group(sampler, builder)
    assert set(group.braintrust_span_exports) == {
        trace.id for trace in harness.traces if not trace.has_error
    }
    assert attempts == 4
    # The failed sibling is excluded from the mean: (1 + 0) / 2 = 0.5.
    # Two and four sampled tokens cost 0.01 and 0.02 before advantages are centered.
    assert group.final_rewards_G == pytest.approx([0.99, -0.02])
    assert [metrics["total_length_penalty"] for metrics in group.metrics_G] == pytest.approx(
        [0.01, 0.02]
    )
    data, _ = assemble_training_data([group], compute_advantages([group]))
    assert len(data) == 2
    for datum, completion, advantage in zip(
        data, [[40, 41], [50, 51, 52, 53]], [0.505, -0.505], strict=True,
    ):
        inputs = datum.loss_fn_inputs
        assert inputs["target_tokens"].to_torch().tolist() == [20] + completion
        assert inputs["advantages"].to_torch().tolist() == pytest.approx(
            [0.0] + [advantage] * len(completion)
        )
    assert all(harness.braintrust.active_rollout(t.id) is None for t in harness.traces)


async def test_scope_cancels_orphaned_running_and_queued_rollouts(setup_rollouts, tmp_path):
    sibling_started = asyncio.Event()
    cancelled = asyncio.Event()
    entered = 0

    async def interrupt_first(trace):
        nonlocal entered
        entered += 1
        if entered == 1:
            await sibling_started.wait()
            raise asyncio.CancelledError("remote rollout interrupted")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runner, builder, harness = setup_rollouts(
        group_size=4,
        concurrency=2,
        before_sample=interrupt_first,
    )
    original_rollout = rollouts.do_group_rollout
    with pytest.raises(asyncio.CancelledError):
        async with runner.installed():
            sampler = await sampler_for_batch(_FakeSamplingClient(), 0, tmp_path)
            await asyncio.wait_for(run_group(sampler, builder), 5)
    assert cancelled.is_set()
    assert harness.active == 0
    assert entered < 4
    assert rollouts.do_group_rollout is original_rollout
    assert all(harness.braintrust.active_rollout(t.id) is None for t in harness.traces)


async def test_harness_timeout_rejects_unscored_group_and_closes_spans(setup_rollouts, tmp_path):
    async def hang(trace):
        await asyncio.Event().wait()

    runner, builder, harness = setup_rollouts(before_sample=hang, timeout=0.02)
    with pytest.raises(ValueError, match="No successful rollouts"):
        async with runner.installed():
            sampler = await sampler_for_batch(_FakeSamplingClient(), 0, tmp_path)
            await asyncio.wait_for(run_group(sampler, builder), 5)
    assert harness.active == 0
    assert all(t.stop_condition == "harness_timeout" for t in harness.traces)
    assert all(harness.braintrust.active_rollout(t.id) is None for t in harness.traces)


async def test_scope_drains_new_tasks_and_preserves_existing_tasks(setup_rollouts):
    runner, _, _ = setup_rollouts()
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def background_work():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    existing = asyncio.create_task(asyncio.Event().wait())
    try:
        async with runner.installed():
            background = asyncio.create_task(background_work())
            await started.wait()
        assert background.cancelled()
        assert cleaned_up.is_set()
        assert not existing.done()
    finally:
        existing.cancel()
        await asyncio.gather(existing, return_exceptions=True)
