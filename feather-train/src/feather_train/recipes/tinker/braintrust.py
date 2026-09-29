"""Tag the rollout spans consumed by successful Cookbook training steps."""

import logging
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps

import braintrust
from tinker_cookbook.rl import train

from .trajectory import FeatherTrajectoryGroup

logger = logging.getLogger(__name__)


@contextmanager
def track_training_batches():
    # Cookbook 0.5.7 has no post-step callback. Keep the wrapper scoped to the run.
    streaming = train.do_train_step_streaming_and_get_sampling_client

    @wraps(streaming)
    async def streaming_step(config, i_batch, *args, **kwargs):
        result = await streaming(config, i_batch, *args, **kwargs)
        try:
            if result is not None:
                _, _, groups = result
                _publish_training_batch(i_batch, (group.trajectory_group for group in groups))
        except Exception:
            logger.warning("Could not read training batch %d for span tagging", i_batch, exc_info=True)
        return result

    train.do_train_step_streaming_and_get_sampling_client = streaming_step
    try:
        yield
    finally:
        train.do_train_step_streaming_and_get_sampling_client = streaming


def _publish_training_batch(batch: int, groups: Iterable[FeatherTrajectoryGroup]) -> None:
    for group in groups:
        for exported in group.braintrust_span_exports:
            try:
                braintrust.update_span(exported, metadata={"training_batch_id": batch})
            except Exception:
                logger.warning("Could not publish training batch %d", batch, exc_info=True)
