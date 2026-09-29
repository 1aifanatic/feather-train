"""Exact recorded Verifiers tokens converted into Cookbook training trajectories."""

from dataclasses import dataclass

import tinker
import verifiers.v1 as vf
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

from feather_train.errors import NonRetryableHarnessError


@dataclass(kw_only=True)
class FeatherTrajectoryGroup(TrajectoryGroup):
    braintrust_span_exports: list[str]


def convert_traces_to_group(traces: list[vf.Trace]) -> FeatherTrajectoryGroup:
    """Consume rewards already finalized by the native training task's group hook."""
    for trace in traces:
        error = trace.error
        if trace.has_error and error is not None and error.type == "NonRetryableHarnessError":
            raise NonRetryableHarnessError(error.message)
    traces = [trace for trace in traces if not trace.has_error]
    if not traces:
        raise ValueError("No successful rollouts remain in the training group")
    return FeatherTrajectoryGroup(
        braintrust_span_exports=[
            t.info["bt_span_export"] for t in traces if t.info.get("bt_span_export")
        ],
        trajectories_G=[convert_trace_to_trajectory(trace) for trace in traces],
        final_rewards_G=[trace.reward for trace in traces],
        metrics_G=[dict(trace.metrics) for trace in traces],
    )


def convert_trace_to_trajectory(trace: vf.Trace) -> Trajectory:
    sampled_indices = [index for index, node in enumerate(trace.nodes) if node.sampled]
    if not sampled_indices:
        raise NonRetryableHarnessError("Training trace has no sampled responses")
    return Trajectory(
        transitions=[
            _transition_from_node(trace, index, episode_done=index == sampled_indices[-1])
            for index in sampled_indices
        ],
        final_ob=tinker.ModelInput.empty(),
    )


def _transition_from_node(trace: vf.Trace, node_index: int, *, episode_done: bool) -> Transition:
    node = trace.nodes[node_index]
    sampled_tokens = sum(node.mask)
    if not sampled_tokens or sampled_tokens != len(node.logprobs):
        raise NonRetryableHarnessError(
            f"Sampled trace node {node_index} requires nonempty action tokens and aligned logprobs"
        )

    path: list[int] = []
    current = node.parent
    while current is not None:
        path.append(current)
        current = trace.nodes[current].parent
    prompt_ids: list[int] = []
    for index in reversed(path):
        prompt_ids.extend(trace.nodes[index].token_ids)
    # Verifiers commits each sampled node as a context prefix followed by action tokens.
    split = len(node.token_ids) - sampled_tokens
    prompt_ids.extend(node.token_ids[:split])
    completion_ids = node.token_ids[split:]
    if not prompt_ids:
        raise NonRetryableHarnessError(f"Sampled trace node {node_index} has no prompt tokens")
    return Transition(
        ob=tinker.ModelInput.from_ints(prompt_ids),
        ac=TokensWithLogprobs(
            tokens=completion_ids,
            maybe_logprobs=list(node.logprobs),
        ),
        reward=0.0,
        episode_done=episode_done,
        metrics={},
    )
