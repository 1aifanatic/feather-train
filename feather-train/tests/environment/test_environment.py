# ruff: noqa: E402
import asyncio
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")

from feather_train.environment import load_environment
from feather_train.harness import FeatherHarness, FeatherHarnessConfig
from feather_train.observability.braintrust import BraintrustRolloutStart
from feather_train.tasks.taskset import FeatherTaskData


def test_new_environment_loads_without_loading_dataset():
    env = load_environment(hf_dataset="unused/test", hf_revision="a" * 40, enable_braintrust=False)
    assert isinstance(env.harness, FeatherHarness)
    assert env.harness.config.compaction.enabled
    assert env.harness.config.id == "feather_train"
    assert env.taskset.config.id == "feather_train"


@pytest.fixture
def trace():
    task = FeatherTaskData.model_validate(
        {
            "idx": 0,
            "task_id": "test",
            "task_name": "test",
            "world_id": "test",
            "hf_dataset": "unused/test",
            "hf_revision": "a" * 40,
            "prompt": [{"role": "user", "content": "solve"}],
            "rubric": [],
            "rubric_doc": {},
            "expected_deliverable": {},
            "submission_transport": "text_response",
            "grading_evidence_mode": "direct_text",
            "requires_web_search": False,
            "network_mode": "public",
            "mcp_servers": [],
        }
    )
    return vf.Trace(
        task=vf.TraceTask(type="FeatherTask", data=task),
        agent=vf.AgentInfo(model="test", sampling=vf.Sampling()),
    )


async def test_completion_is_stamped_before_braintrust_closes(trace):
    class CompletedHarness(FeatherHarness):
        async def launch(self, *args):
            return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    harness = CompletedHarness(FeatherHarnessConfig(enable_braintrust=False))
    rollout = harness.braintrust.open_rollout(
        BraintrustRolloutStart(
            key=trace.id,
            model="test",
            name="test",
            input="",
            expected="",
            metadata={},
        )
    )
    await harness.run(None, trace, None, "endpoint", "secret", {})
    assert rollout.metadata["stop_condition"] == "agent_completed"
    assert harness.braintrust.active_rollout(trace.id) is None


async def test_timeout_closes_with_final_verifiers_stop_condition(trace):
    class CancelledHarness(FeatherHarness):
        async def launch(self, *args):
            raise asyncio.CancelledError

    harness = CancelledHarness(FeatherHarnessConfig(enable_braintrust=False))
    rollout = harness.braintrust.open_rollout(
        BraintrustRolloutStart(
            key=trace.id,
            model="test",
            name="test",
            input="",
            expected="",
            metadata={},
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await harness.run(None, trace, None, "endpoint", "secret", {})
    assert harness.braintrust.active_rollout(trace.id) is rollout
    trace.stop("harness_timeout")
    await harness.score(trace, None)
    assert rollout.metadata["stop_condition"] == "harness_timeout"
    assert harness.braintrust.active_rollout(trace.id) is None


async def test_compaction_does_not_consume_agent_turn_budget(trace):
    from feather_harness import (
        AssistantMessage,
        FeatherAgent,
        RunRequest,
        Tool,
        ToolCall,
        ToolResult,
        ToolSpec,
        UserMessage,
    )
    from feather_harness.core.model import ModelCall
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.session import RolloutLimits

    env = load_environment(
        hf_dataset="unused/test", hf_revision="a" * 40, max_turns=3, enable_braintrust=False
    )
    limits = RolloutLimits(max_turns=env.config.max_turns)
    calls = []
    completed_tools = []

    class Model:
        async def complete(self, context, *, settings, tools=(), kind="agent"):
            assert limits.reached(trace) is None
            calls.append(kind)
            trace.nodes.append(
                MessageNode(message=vf.AssistantMessage(content=kind), sampled=True)
            )
            assistant = (
                AssistantMessage(content="summary")
                if kind == "compaction"
                else AssistantMessage(
                    tool_calls=(ToolCall(id=str(len(calls)), name="work", arguments="{}"),)
                )
            )
            return ModelCall(assistant=assistant, input_tokens=120000, output_tokens=3)

    async def work(raw):
        completed_tools.append(raw)
        return ToolResult(content="finished")

    result = await FeatherAgent(
        model=Model(),
        tools=(Tool(spec=ToolSpec(name="work", description="work", parameters={}), invoke=work),),
        compaction=env.harness.config.compaction,
    ).run(
        RunRequest(messages=(UserMessage(content="work"),), max_turns=env.harness.config.max_turns)
    )

    assert calls == ["agent", "compaction", "agent", "agent"]
    assert len(completed_tools) == result.num_turns == 3
    assert result.compactions == 1
    assert result.stop_reason == "MaxTurnsExceeded"


@pytest.mark.parametrize("configured", [(), ("read",), ("read", "web_fetch", "web_search")])
@pytest.mark.parametrize("required", [False, True])
def test_task_flag_controls_both_web_tools(configured, required):
    config = FeatherHarnessConfig(tools=configured)
    non_web = tuple(name for name in configured if name == "read")
    assert config.tools_for_task(requires_web_search=required) == (
        (*non_web, "web_search", "web_fetch") if required else non_web
    )


async def test_missing_web_credentials_are_non_retryable_before_sandbox_start(trace, monkeypatch):
    from feather_train.errors import NonRetryableHarnessError

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    trace = vf.Trace(
        task=vf.TraceTask(
            type="FeatherTask",
            data=trace.task.data.model_copy(update={"requires_web_search": True}),
        ),
        agent=trace.agent,
    )
    harness = FeatherHarness(FeatherHarnessConfig(enable_braintrust=False))
    with pytest.raises(NonRetryableHarnessError, match="EXA_API_KEY") as error:
        await harness.launch(
            SimpleNamespace(model="test", sampling=vf.Sampling()),
            trace,
            None,
            "endpoint",
            "secret",
            {},
        )
    # Verifiers matches retry exclusions by exact error type name.
    assert type(error.value) is NonRetryableHarnessError
