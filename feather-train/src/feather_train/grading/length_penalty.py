"""Group length costs from Verifiers traces, shared by training integrations."""

from typing import Self

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from feather_train.errors import NonRetryableHarnessError


class LengthPenaltyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    output_token_reference_length: int = Field(gt=0, strict=True)
    num_output_tokens_weight: float = Field(ge=0)
    output_token_free_allowance: int = Field(default=0, ge=0, strict=True)
    num_returned_tool_tokens_weight: float = Field(default=0, ge=0)
    returned_tool_token_reference_length: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def require_tool_reference(self) -> Self:
        if (
            self.num_returned_tool_tokens_weight > 0
            and self.returned_tool_token_reference_length is None
        ):
            raise ValueError("returned_tool_token_reference_length is required for its penalty")
        return self


def calculate_length_penalties(traces: list[vf.Trace], config: LengthPenaltyConfig) -> list[float]:
    """Return positive deductions and record diagnostics without changing rewards.

    Callers supply a settled group after individual scoring, before advantages.
    Failed rollouts receive no deduction and do not affect the successful group mean.
    Token counts and current rewards come directly from the Verifiers traces.
    The output allowance applies once per trajectory; tool tokens have no allowance.
    """
    # A transient sibling failure must not hide a permanent failure in the group.
    for trace in traces:
        error = trace.errors[-1] if trace.errors else None
        if trace.has_error and error is not None and error.type == "NonRetryableHarnessError":
            raise NonRetryableHarnessError(error.message)
    rewards = [trace.reward for trace in traces if not trace.has_error]
    if not rewards:
        return [0.0] * len(traces)
    scale = max(sum(rewards) / len(rewards), 0.0)
    penalties = []
    for trace in traces:
        if trace.has_error:
            penalties.append(0.0)
            continue
        reward = trace.reward
        # Each graph node contributes once, including summary requests. Replayed
        # context has a false mask; walking every context window would double-charge it.
        sampled_nodes = [node for node in trace.nodes if node.sampled]
        if config.num_output_tokens_weight > 0 and any(
            not node.token_ids or len(node.mask) != len(node.token_ids) for node in sampled_nodes
        ):
            raise NonRetryableHarnessError(
                "Training length penalty requires token IDs and sampling masks from a "
                "renderer client. Use FeatherTask for evaluation without length penalties; "
                "FeatherTrainingTask requires renderer-backed training rollouts."
            )
        output_tokens = sum(sum(node.mask) for node in sampled_nodes)
        charged_output_tokens = max(output_tokens - config.output_token_free_allowance, 0)
        output_penalty = (
            scale
            * config.num_output_tokens_weight
            * (charged_output_tokens / config.output_token_reference_length)
        )
        tool_penalty = 0.0
        if config.num_returned_tool_tokens_weight > 0:
            count = _count_tool_body_tokens(trace)
            tool_penalty = (
                scale
                * config.num_returned_tool_tokens_weight
                * (count / config.returned_tool_token_reference_length)
            )
            trace.record_metrics(
                {
                    "length_penalty_returned_tool_tokens": count,
                    "returned_tool_length_penalty": tool_penalty,
                }
            )
        penalty = output_penalty + tool_penalty
        trace.record_metrics(
            {
                "reward_before_length_penalty": reward,
                "reward_after_length_penalty": reward - penalty,
                "length_penalty_output_tokens": output_tokens,
                "output_length_penalty": output_penalty,
                "total_length_penalty": penalty,
            }
        )
        penalties.append(penalty)
    return penalties


def _count_tool_body_tokens(trace: vf.Trace) -> int:
    """Charge each tool result's first rendered body, excluding compaction replay."""
    seen: set[str] = set()
    count = 0
    for node in trace.nodes:
        if not isinstance(node.message, vf.ToolMessage):
            continue
        call_id = node.message.tool_call_id
        if call_id in seen:
            continue
        if not node.token_ids or len(node.is_content) != len(node.token_ids):
            raise NonRetryableHarnessError(
                "Training tool length penalty requires renderer content-token attribution. "
                "Use FeatherTask for evaluation without length penalties; "
                "FeatherTrainingTask requires a renderer with tool-body attribution."
            )
        seen.add(call_id)
        count += sum(node.is_content)
    # Final-turn results have no graph node; rollout finalization renders them once.
    unsent = trace.info.get("unsent_tool_output_tokens", 0)
    if unsent is None:
        raise NonRetryableHarnessError(
            "Training tool length penalty requires renderer content attribution for "
            "the final unsent tool results"
        )
    return count + unsent
