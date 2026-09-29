from __future__ import annotations

from types import SimpleNamespace

import verifiers.v1 as vf
from renderers import (
    ParsedResponse,
    RenderedTokens,
    RendererPool,
)

from feather_train.clients.tinker import TinkerClient


class _FakeRenderer:
    supports_tools = True

    def __init__(self, rendered_ids: list[int]):
        self.rendered_ids = rendered_ids
        self.render_calls: list[dict] = []
        self.bridge_calls: list[dict] = []
        self.bridge_result: RenderedTokens | None = RenderedTokens(token_ids=[10, 20, 21, 30, 31])
        self.parsed = ParsedResponse(content="completion", tool_calls=[])

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        self.render_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return RenderedTokens(token_ids=list(self.rendered_ids))

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[dict],
        *,
        tools=None,
    ):
        self.bridge_calls.append(
            {
                "previous_prompt_ids": previous_prompt_ids,
                "previous_completion_ids": previous_completion_ids,
                "new_messages": new_messages,
                "tools": tools,
            }
        )
        return self.bridge_result

    def parse_response(self, token_ids: list[int], *, tools=None):
        return self.parsed

    def get_stop_token_ids(self) -> list[int]:
        return [99]


class _FakeSamplingClient:
    def __init__(
        self,
        completion_ids: list[int] | None = None,
        error: BaseException | None = None,
    ):
        self.completion_ids = completion_ids or [40, 50]
        self.logprobs: list[float] | None = [-0.25] * len(self.completion_ids)
        self.error = error
        self.requests: list[dict] = []

    async def sample_async(self, *, prompt, num_samples: int, sampling_params):
        self.requests.append(
            {
                "prompt": prompt,
                "num_samples": num_samples,
                "sampling_params": sampling_params,
            }
        )
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=list(self.completion_ids),
                    logprobs=self.logprobs,
                    stop_reason="stop",
                )
            ]
        )


def _make_client(
    *,
    renderer: _FakeRenderer | None = None,
    max_context_length: int | None = None,
) -> tuple[TinkerClient, _FakeRenderer, _FakeSamplingClient]:
    fake_renderer = renderer or _FakeRenderer(rendered_ids=[10, 20])
    sampling_client = _FakeSamplingClient()
    return (
        TinkerClient(
            sampling_client=sampling_client,
            renderer=RendererPool(lambda: fake_renderer, size=1),
            max_context_length=max_context_length,
        ),
        fake_renderer,
        sampling_client,
    )


def _trace_with_prior_turn() -> tuple[vf.Trace, list[vf.Message]]:
    user = vf.UserMessage(content="hello")
    assistant = vf.AssistantMessage(content="first")
    next_user = vf.UserMessage(content="next")
    task = vf.TaskData(idx=0, name="task", prompt=[user])
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            vf.MessageNode(
                message=user,
                token_ids=[10, 20],
                mask=[False, False],
            ),
            vf.MessageNode(
                parent=0,
                message=assistant,
                sampled=True,
                token_ids=[21, 30],
                mask=[False, True],
                logprobs=[-0.1],
            ),
        ],
    )
    return trace, [user, assistant, next_user]


def _body(messages: list[dict] | None = None, *, tools: list[dict] | None = None) -> dict:
    body: dict = {"messages": messages or [{"role": "user", "content": "hello"}]}
    if tools is not None:
        body["tools"] = tools
    return body
