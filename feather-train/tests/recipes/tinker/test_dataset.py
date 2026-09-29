import pytest
import verifiers.v1 as vf
from tinker_cookbook.utils.ml_log import dump_config

from feather_train.grading.length_penalty import LengthPenaltyConfig
from feather_train.recipes.tinker.dataset import FeatherDataset, FeatherDatasetBuilder
from feather_train.tasks.taskset import FeatherTask, FeatherTaskData


@pytest.fixture
def penalty():
    return LengthPenaltyConfig(output_token_reference_length=10, num_output_tokens_weight=0.1)


async def test_batches_preserve_task_order_and_exclude_incomplete_final_batch(penalty):
    dataset, _ = await FeatherDatasetBuilder(
        dataset=FeatherDataset([make_task(i) for i in [4, 1, 7, 3, 9]], penalty, groups_per_batch=2)
    )()
    batches = [dataset.get_batch(index) for index in range(len(dataset))]
    assert [[builder.task.data.task_id for builder in batch] for batch in batches] == [
        ["4", "1"], ["7", "3"],
    ]
    assert batches[0][0].task.data.rollout_group_id == "group-4"


def test_cookbook_config_logging_omits_loaded_task_contents(penalty):
    builder = FeatherDatasetBuilder(
        dataset=FeatherDataset([make_task(i) for i in range(5)], penalty, groups_per_batch=2)
    )
    assert dump_config(builder) == {"dataset": {"num_tasks": 5, "groups_per_batch": 2}}


def test_training_task_keeps_configured_judges(penalty):
    task = make_task(0)
    task.config = vf.TaskConfig(judges=[vf.JudgeConfig(id="reference", name="extra_judge")])
    dataset = FeatherDataset([task], penalty, groups_per_batch=1)
    assert dataset.get_batch(0)[0].task.config.judges == task.config.judges


async def test_unshaped_singleton_does_not_require_renderer_tokens():
    task = FeatherDataset([make_task(0)], None, groups_per_batch=1).get_batch(0)[0].task
    trace = vf.Trace(
        task=vf.TraceTask(type="FeatherTask", data=task.data), info={"grading_score": 0.8}
    )
    await task.score(trace)
    await task.score_group([trace])
    assert trace.reward == 0.8
    assert "length_penalty" not in trace.rewards


def make_task(index):
    return FeatherTask(
        FeatherTaskData(
            idx=index,
            prompt="solve",
            task_id=str(index),
            task_name="Task",
            world_id="world",
            requires_web_search=False,
            network_mode="no-network",
            hf_dataset="test/tasks",
            hf_revision="a" * 40,
            mcp_servers=[],
            rollout_group_id=f"group-{index}",
        )
    )
