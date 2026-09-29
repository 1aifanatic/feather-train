import pytest
import verifiers.v1 as vf
from renderers import RendererPool
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.graph import prepare_turn

from feather_train.environment import load_environment
from feather_train.grading.length_penalty import LengthPenaltyConfig
from feather_train.harness import FeatherHarness, FeatherHarnessConfig
from feather_train.observability.braintrust import BraintrustRolloutStart
from feather_train.recipes.tinker.dataset import FeatherGroupBuilder
from feather_train.recipes.tinker.rollout import FeatherRollouts
from feather_train.tasks.taskset import FeatherTaskData
from feather_train.tasks.training import FeatherTrainingTask, FeatherTrainingTaskConfig

from ...clients.helpers import _FakeRenderer


class LocalHarness(FeatherHarness):
    def __init__(self, before_sample=None):
        super().__init__(FeatherHarnessConfig(enable_braintrust=False))
        self.before_sample = before_sample
        self.traces = []
        self.active = 0
        self.peak = 0

    async def launch(self, ctx, trace, *args):
        self.traces.append(trace)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.braintrust.open_rollout(
            BraintrustRolloutStart(
                key=trace.id,
                model=ctx.model,
                name="test",
                input="solve",
                expected=None,
                metadata={},
            )
        )
        try:
            if self.before_sample:
                await self.before_sample(trace)
            turn = prepare_turn(trace, [vf.UserMessage(content="solve")])
            response = await ctx.client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": "solve"}]},
                ctx.model,
                ctx.sampling,
                turn=turn,
            )
            turn.commit(response)
            trace.info.setdefault("grading_score", 1.0)
            return vf.ProgramResult(exit_code=0, stdout="", stderr="")
        finally:
            self.active -= 1


@pytest.fixture
def setup_rollouts():
    def make(*, group_size=2, concurrency=2, retries=1, before_sample=None, timeout=None):
        env = load_environment(
            hf_dataset="test/tasks",
            hf_revision="a" * 40,
            rollout_max_retries=retries,
            enable_braintrust=False,
        )
        env.harness = LocalHarness(before_sample)
        runner = FeatherRollouts(
            env,
            RendererPool(lambda: _FakeRenderer([10, 20]), size=1),
            model="test-model",
            group_size=group_size,
            concurrency=concurrency,
            max_context_length=10,
            top_p=0.8,
            top_k=12,
        )
        task = FeatherTrainingTask(
            FeatherTaskData(
                idx=0,
                prompt="solve",
                task_id="task",
                task_name="Task",
                world_id="world",
                timeout={"harness": timeout},
                requires_web_search=False,
                network_mode="no-network",
                hf_dataset="test/tasks",
                hf_revision="a" * 40,
                mcp_servers=[],
            ),
            FeatherTrainingTaskConfig(
                length_penalty=LengthPenaltyConfig(
                    output_token_reference_length=10,
                    num_output_tokens_weight=0.1,
                )
            ),
        )
        return runner, FeatherGroupBuilder(task), env.harness

    return make

