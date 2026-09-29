"""Verifiers episodes at Cookbook's group-rollout boundary."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import cast
from weakref import WeakKeyDictionary

import braintrust
import tinker
import verifiers.v1 as vf
from renderers import RendererPool
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.rl import rollouts, train
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.rollout_strategy import RolloutStrategy
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup

from feather_train.clients.tinker import TinkerClient

from .braintrust import track_training_batches
from .dataset import FeatherGroupBuilder
from .trajectory import FeatherTrajectoryGroup, convert_traces_to_group

logger = logging.getLogger(__name__)


class FeatherRollouts:
    def __init__(
        self,
        env: vf.Environment,
        renderer: RendererPool,
        *,
        model: str,
        group_size: int,
        concurrency: int,
        max_context_length: int,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> None:
        self.env = env
        self.renderer = renderer
        self.model = model
        self.group_size = group_size
        self.max_context_length = max_context_length
        self.top_p = top_p
        self.top_k = top_k
        self.policy_versions: WeakKeyDictionary[tinker.SamplingClient, int] = WeakKeyDictionary()
        self.initial_sampler_step: int | None = None
        self.rollout_semaphore = asyncio.Semaphore(concurrency)
        self.episodes: set[vf.Episode] = set()

    @asynccontextmanager
    async def installed(self):
        """Own background tasks for one standalone training run on this event loop."""
        original_rollout = rollouts.do_group_rollout
        original_sampler = train.save_checkpoint_and_get_sampling_client
        original_async_training = train.do_async_training
        existing_tasks = asyncio.all_tasks()

        async def save_sampler(training_client, checkpoint_mgr, i_batch, start_batch=0):
            client, metrics = await original_sampler(
                training_client, checkpoint_mgr, i_batch, start_batch
            )
            self.policy_versions[client] = i_batch
            return client, metrics

        async def async_training(start_batch: int, end_batch: int, **kwargs) -> None:
            # Async Cookbook creates its initial sampler outside the sampler helper.
            self.initial_sampler_step = start_batch
            await original_async_training(start_batch=start_batch, end_batch=end_batch, **kwargs)

        rollouts.do_group_rollout = self.run_group
        train.save_checkpoint_and_get_sampling_client = save_sampler
        train.do_async_training = async_training
        with track_training_batches():
            failure: BaseException = asyncio.CancelledError("Training ended")
            try:
                yield
            except BaseException as exc:
                failure = exc
                raise
            finally:
                try:
                    await self._drain(failure, existing_tasks)
                finally:
                    rollouts.do_group_rollout = original_rollout
                    train.save_checkpoint_and_get_sampling_client = original_sampler
                    train.do_async_training = original_async_training

    async def run_group(
        self,
        builder: EnvGroupBuilder,
        policy: TokenCompleter,
        strategy: RolloutStrategy | None = None,
        termination: TerminationRewardPolicy | None = None,
    ) -> TrajectoryGroup:
        # Verifiers owns retries and reward shaping; Cookbook supplies these kwargs.
        del strategy, termination
        policy = cast(TinkerTokenCompleter, policy)
        if (
            policy.sampling_client not in self.policy_versions
            and self.initial_sampler_step is not None
        ):
            self.policy_versions[policy.sampling_client] = self.initial_sampler_step
            self.initial_sampler_step = None
        task = cast(FeatherGroupBuilder, builder).task
        task = type(task)(
            task.data.model_copy(
                update={"sampled_policy_version": self.policy_versions[policy.sampling_client]}
            ),
            task.config,
        )
        client = self.create_client(policy.sampling_client)
        ctx = vf.ModelContext(
            model=self.model,
            client=client,
            sampling=vf.Sampling(
                max_tokens=policy.max_tokens,
                temperature=policy.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )
        episode = self.env.episode(task, ctx, n=self.group_size)
        self.episodes.add(episode)
        traces = await episode.run(semaphore=self.rollout_semaphore)
        self.episodes.remove(episode)
        result = self.convert_group(traces)
        _publish_group_rewards(traces)
        return result

    def create_client(self, sampling_client: tinker.SamplingClient) -> vf.Client:
        return TinkerClient(
            sampling_client=sampling_client,
            renderer=self.renderer,
            max_context_length=self.max_context_length,
        )

    def convert_group(self, traces: list[vf.Trace]) -> FeatherTrajectoryGroup:
        return convert_traces_to_group(traces)

    async def _drain(self, error: BaseException, existing_tasks: set[asyncio.Task]) -> None:
        # This recipe owns tasks started during training; services already running
        # at entry are closed separately by their own context managers.
        pending = asyncio.all_tasks() - existing_tasks
        for task in pending:
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for episode in self.episodes:
            for rollout in episode.rollouts:
                if rollout.trace is not None:
                    try:
                        await self.env.harness.finalize_interrupted_trace(rollout.trace, error)
                    except Exception:
                        logger.warning(
                            "Could not finalize interrupted training trace", exc_info=True
                        )
        self.episodes.clear()


def _publish_group_rewards(traces: list[vf.Trace]) -> None:
    # The harness closes rollout spans before Verifiers applies group rewards.
    for trace in traces:
        exported = trace.info.get("bt_span_export")
        if exported:
            try:
                braintrust.update_span(
                    exported,
                    metrics={**trace.metrics, "reward": trace.reward},
                )
            except Exception:
                logger.warning("Could not publish group rewards for %s", trace.id, exc_info=True)
