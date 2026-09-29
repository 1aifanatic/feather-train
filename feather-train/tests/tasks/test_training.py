import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import prepare_turn

from feather_train.errors import NonRetryableHarnessError
from feather_train.grading.length_penalty import LengthPenaltyConfig
from feather_train.tasks.taskset import FeatherTask, FeatherTaskData
from feather_train.tasks.training import (
    FeatherTrainingTask,
    FeatherTrainingTaskConfig,
)


@pytest.fixture
def task():
    data = FeatherTaskData(
        idx=0,
        prompt="Solve the task",
        task_id="task",
        task_name="Task",
        world_id="world",
        requires_web_search=False,
        network_mode="no-network",
        hf_dataset="test/tasks",
        hf_revision="a" * 40,
        mcp_servers=[],
    )
    return FeatherTrainingTask(
        data,
        FeatherTrainingTaskConfig(
            length_penalty=LengthPenaltyConfig(
                output_token_reference_length=10,
                num_output_tokens_weight=0.1,
            )
        ),
    )


def trace(task, reward, output_tokens):
    return vf.Trace(
        task=vf.TraceTask(type="FeatherTrainingTask", data=task.data),
        info={"grading_score": reward},
        nodes=[
            vf.MessageNode(
                message=vf.AssistantMessage(content="answer"),
                sampled=True,
                token_ids=list(range(output_tokens)),
                mask=[True] * output_tokens,
                logprobs=[-0.1] * output_tokens,
            )
        ],
    )


async def test_group_hook_uses_mean_reward_and_preserves_grader_score(task):
    traces = [trace(task, 1.0, 10), trace(task, 0.0, 20)]
    for item in traces:
        await task.score(item)
    await task.score_group(traces)

    assert [item.reward for item in traces] == pytest.approx([0.95, -0.1])
    assert [item.rewards["harbor_grader_reward"] for item in traces] == [1.0, 0.0]
    assert [item.rewards["length_penalty"] for item in traces] == pytest.approx([-0.05, -0.1])
    assert [item.metrics["reward_after_length_penalty"] for item in traces] == pytest.approx(
        [0.95, -0.1]
    )


async def test_failed_rollout_is_excluded_from_length_penalty_and_group_mean(task):
    traces = [trace(task, 1.0, 10), trace(task, 9.0, 0), trace(task, 0.0, 20)]
    for item in traces:
        await task.score(item)
    traces[1].errors.append(vf.Error(type="HarnessError", message="sandbox failed"))
    await task.score_group(traces)
    assert [item.rewards["length_penalty"] for item in traces] == pytest.approx([-0.05, 0, -0.1])
    assert [item.reward for item in traces] == pytest.approx([0.95, 9.0, -0.1])
    assert "length_penalty_output_tokens" not in traces[1].metrics


async def test_all_failed_rollouts_receive_no_length_penalty(task):
    traces = [trace(task, 0.0, 0), trace(task, 0.0, 0)]
    for item in traces:
        item.errors.append(vf.Error(type="HarnessError", message="retries exhausted"))
    await task.score_group(traces)
    assert [item.rewards["length_penalty"] for item in traces] == [0, 0]


async def test_fatal_rollout_error_survives_group_scoring(task):
    traces = [trace(task, 0.0, 10), trace(task, 0.0, 20)]
    traces[0].errors.append(vf.Error(type="ProviderError", message="temporary outage"))
    traces[1].errors.append(vf.Error(type="NonRetryableHarnessError", message="broken contract"))
    with pytest.raises(NonRetryableHarnessError, match="broken contract"):
        await task.score_group(traces)
    assert all("length_penalty" not in item.rewards for item in traces)


async def test_tokenless_api_response_is_valid_for_eval_but_not_training(task):
    item = vf.Trace(
        task=vf.TraceTask(type="FeatherTask", data=task.data),
        info={"grading_score": 0.8},
    )
    turn = prepare_turn(item, [vf.UserMessage(content="Solve the task")])
    turn.commit(
        vf.Response(
            id="eval-response",
            created=0,
            model="api-model",
            message=vf.AssistantMessage(content="The answer"),
            finish_reason="stop",
        )
    )
    eval_task = FeatherTask(task.data)
    await eval_task.score(item)
    await eval_task.score_group([item])
    assert item.rewards == {"harbor_grader_reward": 0.8}
    with pytest.raises(NonRetryableHarnessError, match="Use FeatherTask for evaluation"):
        await task.score_group([item])
    assert item.rewards == {"harbor_grader_reward": 0.8}
