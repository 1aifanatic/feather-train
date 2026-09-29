import pytest
import verifiers.v1 as vf
from pydantic import ValidationError

from feather_train.errors import NonRetryableHarnessError
from feather_train.grading.length_penalty import LengthPenaltyConfig, calculate_length_penalties


@pytest.fixture
def config():
    return LengthPenaltyConfig(
        output_token_reference_length=10,
        num_output_tokens_weight=0.1,
    )


@pytest.fixture
def tool_config(config):
    return config.model_copy(update={
        "returned_tool_token_reference_length": 10,
        "num_returned_tool_tokens_weight": 0.2,
    })


def test_shared_calculation_returns_deductions_without_changing_rewards(tool_config):
    traces = [trace(10, 1.0), trace(20, 0.0)]
    for item in traces:
        item.nodes.append(tool_node("read"))
    original_rewards = [dict(item.rewards) for item in traces]

    assert calculate_length_penalties(traces, tool_config) == pytest.approx([0.07, 0.12])
    assert [item.rewards for item in traces] == original_rewards
    assert [item.metrics["length_penalty_returned_tool_tokens"] for item in traces] == [2, 2]


@pytest.mark.parametrize(
    ("allowance", "output_penalty"), [(0, 0.07), (6, 0.01), (7, 0.0), (8, 0.0)]
)
def test_output_allowance_applies_once_across_messages_without_discounting_tools(
    tool_config, allowance, output_penalty
):
    item = trace(3, 1.0)
    item.nodes.extend([sampled_node(4), tool_node("read")])
    config = tool_config.model_copy(update={"output_token_free_allowance": allowance})

    assert calculate_length_penalties([item], config) == pytest.approx([output_penalty + 0.04])
    assert item.metrics["length_penalty_output_tokens"] == 7
    assert item.metrics["output_length_penalty"] == pytest.approx(output_penalty)
    assert item.metrics["returned_tool_length_penalty"] == pytest.approx(0.04)


def test_summary_tokens_count_once_and_replayed_context_is_free(config):
    item = trace(10, 1.0)
    item.nodes.extend([
        sampled_node(2, parent=0),
        vf.MessageNode(
            message=vf.UserMessage(content="summary and retained context"),
            token_ids=list(range(12)),
            mask=[False] * 12,
        ),
        vf.MessageNode(
            parent=2,
            message=vf.AssistantMessage(content="continued"),
            sampled=True,
            token_ids=[30, 31, 32, 33],
            mask=[False, True, True, True],
            logprobs=[-0.1] * 3,
        ),
    ])

    assert calculate_length_penalties([item], config) == pytest.approx([0.15])
    assert item.metrics["length_penalty_output_tokens"] == 15


@pytest.mark.parametrize("rewards", [[0.0, 0.0], [-1.0, 0.5]])
def test_nonpositive_group_mean_does_not_reward_length(config, rewards):
    assert calculate_length_penalties([trace(10, reward) for reward in rewards], config) == [0, 0]


def test_tool_cost_counts_rendered_bodies_once_per_call(tool_config):
    item = trace(10, 1.0)
    item.nodes.extend([tool_node("a"), tool_node("a"), tool_node("b")])
    # Raw-text accounting is deliberately irrelevant to this reward.
    item.record_metric("returned_tool_tokens", 999)

    assert calculate_length_penalties([item], tool_config) == pytest.approx([0.18])
    assert item.metrics["length_penalty_returned_tool_tokens"] == 4
    assert item.metrics["output_length_penalty"] == pytest.approx(0.1)
    assert item.metrics["returned_tool_length_penalty"] == pytest.approx(0.08)


def test_unsent_token_count_adds_to_observed_tool_cost(tool_config):
    item = trace(10, 1.0)
    item.nodes.append(tool_node("earlier"))
    item.info["unsent_tool_output_tokens"] = 3

    assert calculate_length_penalties([item], tool_config) == pytest.approx([0.2])
    assert item.metrics["length_penalty_returned_tool_tokens"] == 5


def test_tool_cost_requires_content_attribution_only_when_enabled(config, tool_config):
    item = trace(10, 1.0)
    node = tool_node("a")
    node.is_content = []
    item.nodes.append(node)
    with pytest.raises(NonRetryableHarnessError, match="content-token attribution"):
        calculate_length_penalties([item], tool_config)
    assert calculate_length_penalties([item], config) == pytest.approx([0.1])


def test_tool_cost_requires_its_reference_length():
    with pytest.raises(ValidationError, match="returned_tool_token_reference_length"):
        LengthPenaltyConfig(
            output_token_reference_length=10,
            num_output_tokens_weight=0.1,
            num_returned_tool_tokens_weight=0.2,
        )


def test_missing_unsent_attribution_matters_only_when_tool_cost_enabled(config, tool_config):
    item = trace(10, 1.0)
    item.info["unsent_tool_output_tokens"] = None
    assert calculate_length_penalties([item], config) == pytest.approx([0.1])
    with pytest.raises(NonRetryableHarnessError, match="final unsent tool results"):
        calculate_length_penalties([item], tool_config)


def trace(output_tokens, reward):
    item = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="task")),
        nodes=[sampled_node(output_tokens)],
    )
    item.record_reward("grade", reward)
    return item


def sampled_node(count, *, parent=None):
    return vf.MessageNode(
        parent=parent,
        message=vf.AssistantMessage(content="answer"),
        sampled=True,
        token_ids=list(range(count)),
        mask=[True] * count,
        logprobs=[-0.1] * count,
    )


def tool_node(call_id):
    return vf.MessageNode(
        message=vf.ToolMessage(tool_call_id=call_id, content="same text"),
        token_ids=[100, 101, 102, 103],
        mask=[False] * 4,
        is_content=[False, True, True, False],
    )
