from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from renderers import (
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    RendererPool,
    ToolCallParseStatus,
)
from tinker_cookbook.rl.data_processing import trajectory_to_data
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.graph import prepare_turn

from feather_train.clients.tinker import TinkerClient
from feather_train.errors import NonRetryableHarnessError
from feather_train.recipes.tinker.trajectory import (
    convert_trace_to_trajectory,
    convert_traces_to_group,
)


async def test_cookbook_packing_preserves_tool_turns_summary_and_compacted_context():
    renderer = ScriptedRenderer()
    sampler = ScriptedSampler()
    client = TinkerClient(sampling_client=sampler, renderer=RendererPool(lambda: renderer, size=1))
    item = trace()
    async def sample(messages, prompt_ids, message_indices, completion_ids, *, tool_calls=()):
        renderer.prompt = RenderedTokens(token_ids=prompt_ids, message_indices=message_indices)
        renderer.response = ParsedResponse(
            content=" ".join(map(str, completion_ids)), tool_calls=list(tool_calls),
        )
        sampler.tokens = completion_ids
        turn = prepare_turn(item, messages)
        response = await client.get_response(
            ChatDialect(),
            {"tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}]},
            "test-policy", vf.Sampling(max_tokens=8), turn=turn,
        )
        turn.commit(response)
        return response.message

    user = vf.UserMessage(content="solve")
    call = await sample(
        [user], [1, 2, 3], [0, 0, -1], [10, 11],
        tool_calls=[ParsedToolCall(
            raw="read", name="read", arguments="{}", status=ToolCallParseStatus.OK,
        )],
    )
    tool = vf.ToolMessage(tool_call_id=call.tool_calls[0].id, content="result")
    # Bridge attribution covers only the new tool result; prior tokens are reused.
    reply = await sample(
        [user, call, tool],
        [1, 2, 3, 10, 11, 4, 5, 6], [-1] * 5 + [0, 0, -1], [12],
    )
    summary = await sample(
        [vf.SystemMessage(content="Summarize the history"), user, call, tool, reply],
        [7, 1, 2, 3, 10, 11, 4, 5, 6, 12, 8], [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, -1], [20, 21],
    )
    # Summary tokens appear again as context after compaction.
    await sample(
        [vf.SystemMessage(content="Policy instructions"), vf.UserMessage(content=summary.content)],
        [9, 20, 21, 8], [0, 1, 1, -1], [30],
    )

    assert item.info["tito_bridge_modes"] == ["initial", "incremental", "fallback", "fallback"]
    trajectory = convert_trace_to_trajectory(item)
    assert [step.ob.to_ints() for step in trajectory.transitions] == sampler.prompts
    assert [step.episode_done for step in trajectory.transitions] == [False, False, False, True]
    datums = trajectory_to_data(trajectory, 0.75)
    sequences = [
        [1, 2, 3, 10, 11, 4, 5, 6, 12],
        [7, 1, 2, 3, 10, 11, 4, 5, 6, 12, 8, 20, 21],
        [9, 20, 21, 8, 30],
    ]
    masks = [[0, 0, 1, 1, 0, 0, 0, 1], [0] * 10 + [1, 1], [0, 0, 0, 1]]
    trained_tokens = []
    for datum, sequence, mask in zip(datums, sequences, masks, strict=True):
        inputs = datum.loss_fn_inputs
        assert datum.model_input.to_ints() == sequence[:-1]
        assert inputs["target_tokens"].data == sequence[1:]
        assert inputs["mask"].data == mask
        assert inputs["advantages"].data == [value * 0.75 for value in mask]
        assert inputs["logprobs"].data == pytest.approx([
            -token / 100 if sampled else 0
            for token, sampled in zip(sequence[1:], mask, strict=True)
        ])
        trained_tokens.extend(
            token for token, sampled in zip(sequence[1:], mask, strict=True) if sampled
        )
    assert trained_tokens == [10, 11, 12, 20, 21, 30]


def test_tokenless_response_cannot_become_a_training_sample():
    item = trace()
    prepare_turn(item, [vf.UserMessage(content="solve")]).commit(vf.Response(
        id="response", created=0, model="api-model", finish_reason="stop",
        message=vf.AssistantMessage(content="answer"),
    ))
    with pytest.raises(NonRetryableHarnessError, match="nonempty action tokens"):
        convert_trace_to_trajectory(item)


def test_trace_without_samples_cannot_silently_train_an_empty_rollout():
    with pytest.raises(NonRetryableHarnessError, match="no sampled responses"):
        convert_trace_to_trajectory(trace())


def trace():
    return vf.Trace(task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="task")))


class ScriptedRenderer:
    supports_tools = True

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        return self.prompt

    def bridge_to_next_turn(self, previous_prompt_ids, previous_completion_ids, messages, *, tools=None):
        return self.prompt

    def parse_response(self, token_ids, *, tools=None):
        return self.response

    def get_stop_token_ids(self):
        return [99]


class ScriptedSampler:
    def __init__(self):
        self.prompts = []

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.prompts.append(prompt.to_ints())
        return SimpleNamespace(sequences=[SimpleNamespace(
            tokens=self.tokens, logprobs=[-token / 100 for token in self.tokens], stop_reason="stop",
        )])



def test_permanent_failure_cannot_be_dropped_from_an_unshaped_group():
    completed = trace()
    failed = trace()
    failed.errors = [vf.Error(type="NonRetryableHarnessError", message="invalid sandbox")]
    with pytest.raises(NonRetryableHarnessError, match="invalid sandbox"):
        convert_traces_to_group([completed, failed])
