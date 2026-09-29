from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verifiers.v1 import Trace

from feather_train.observability.braintrust import BraintrustRolloutStart

if TYPE_CHECKING:
    from feather_train.tasks.taskset import FeatherTaskData

OBSERVATION_SCHEMA_VERSION = 1


def build_braintrust_rollout_start(
    task: FeatherTaskData,
    trace: Trace,
    *,
    model_str: str,
    max_turns: int,
    tools: list[str] | None = None,
) -> BraintrustRolloutStart:
    info = trace.info
    rollout_group_id = info.get("rollout_group_id")
    source_metadata = task.source_metadata
    eval_run_id = info.get("eval_run_id", task.eval_run_id)
    metadata: dict[str, Any] = {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "eval_run_id": eval_run_id,
        "agent_env": "FeatherEnv",
        "model": model_str,
        "max_turns": max_turns,
        "hf_dataset": task.hf_dataset,
        "hf_revision": task.hf_revision,
        "dataset_id": source_metadata.dataset_id or task.hf_dataset,
        "dataset_revision": source_metadata.dataset_revision or task.hf_revision,
        "source_job_id": source_metadata.source_job_id,
        "query_id": source_metadata.query_id,
        "content_hash": source_metadata.content_hash,
        "rollout_group_id": rollout_group_id,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "world_id": task.world_id,
        "trajectory_id": trace.id,
        "domain": task.domain,
        "eval": eval_run_id is not None or bool(info.get("eval", False)),
    }
    # Experiments that differ only in tooling or config are otherwise indistinguishable on the
    # root span.
    if tools is not None:
        metadata["tools"] = list(tools)
    sampled_policy_version = info.get("sampled_policy_version")
    if sampled_policy_version is not None:
        metadata["sampled_policy_version"] = sampled_policy_version
    return BraintrustRolloutStart(
        key=trace.id,
        model=model_str,
        name=f"{task.task_id}/{task.world_id}/{trace.id}",
        input=task.prompt,
        expected=task.rubric,
        metadata=metadata,
    )


def braintrust_close_metrics(
    trace: Trace,
    *,
    grading_score: float | None,
) -> dict[str, int | float]:
    metrics = _tito_bridge_metrics(trace)
    metrics.update(trace.metrics)
    if grading_score is not None:
        metrics["reward"] = float(trace.info.get("training_reward", grading_score))
    return metrics


def _tito_bridge_metrics(trace: Trace) -> dict[str, int]:
    bridge_counts = {"initial": 0, "incremental": 0, "fallback": 0}
    for bridge_mode in trace.info.get("tito_bridge_modes") or []:
        if bridge_mode in bridge_counts:
            bridge_counts[bridge_mode] += 1
    return {
        f"tito_bridge_{bridge_mode}_count": count
        for bridge_mode, count in bridge_counts.items()
        if count
    }
