import verifiers.v1 as vf
from feather_harness.core.messages import AssistantMessage, ToolCall, ToolMessage
from feather_harness.core.runner import RunResult, TraceSegment
from renderers import RenderedTokens, RendererPool

from feather_train.clients.tinker import TinkerClient
from feather_train.grading.unsent_tool_results import (
    count_tool_content_tokens,
    get_unsent_tool_turn,
)
from feather_train.harness import FeatherHarness, FeatherHarnessConfig


def test_turn_limit_extracts_only_final_batch_with_its_call_context():
    call = AssistantMessage(tool_calls=(
        ToolCall(id="a", name="read", arguments='{"path":"a"}'),
        ToolCall(id="b", name="read", arguments='{"path":"b"}'),
    ))
    result = RunResult(
        stop_reason="MaxTurnsExceeded", num_turns=2, compactions=0,
        segments=(TraceSegment(kind="execution", input=(), messages=(
            ToolMessage(tool_call_id="earlier", content="already observed"),
            call,
            ToolMessage(tool_call_id="a", content="first body"),
            ToolMessage(tool_call_id="b", content="second body"),
        )),),
    )
    turn = get_unsent_tool_turn(result)
    assert [message["role"] for message in turn] == ["assistant", "tool", "tool"]
    assert [message["content"] for message in turn[1:]] == ["first body", "second body"]
    assert turn[0]["tool_calls"][0]["function"]["name"] == "read"
    assert get_unsent_tool_turn(result.model_copy(update={"stop_reason": "completed"})) == []


def test_body_count_excludes_assistant_and_template_tokens():
    rendered = RenderedTokens(
        token_ids=list(range(7)),
        message_indices=[0, 0, 1, 1, 1, 2, 2],
        message_roles=["assistant", "tool", "tool"],
        is_content=[False, True, False, True, True, False, True],
    )
    assert count_tool_content_tokens(rendered) == 3


def test_missing_body_attribution_is_distinct_from_empty_tool_content():
    assert count_tool_content_tokens(RenderedTokens(token_ids=[1, 2])) is None


async def test_harness_counts_tool_content_without_sampling():
    class Renderer:
        def render(self, messages, *, add_generation_prompt):
            assert not add_generation_prompt
            assert messages == [{"role": "tool", "content": "hello"}]
            return RenderedTokens(
                token_ids=[1, 2, 3], message_indices=[0, 0, 0],
                message_roles=["tool"], is_content=[False, True, False],
            )

    class Sampler:
        async def sample_async(self, **kwargs):
            raise AssertionError("Counting must not request model inference")

    client = TinkerClient(sampling_client=Sampler(), renderer=RendererPool(Renderer, size=1))
    harness = FeatherHarness(FeatherHarnessConfig(enable_braintrust=False))
    assert await harness.count_tool_output_tokens(
        vf.ModelContext(model="test", client=client, sampling=vf.Sampling()), [{"role": "tool", "content": "hello"}]
    ) == 1
