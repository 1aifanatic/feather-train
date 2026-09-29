"""Nonstreaming Chat Completions using the caller's client and retry policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, JsonValue

from ..tools import ToolSpec
from .messages import (
    AssistantMessage,
    FrozenModel,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class ModelBehaviorError(ValueError):
    pass


class ModelRefusalError(RuntimeError):
    pass


class ModelSettings(FrozenModel):
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    top_logprobs: int | None = None
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)


class ModelCall(FrozenModel):
    assistant: AssistantMessage
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None


class ChatCompletionsModel:
    def __init__(self, *, client, model: str, replay_reasoning: bool = True):
        self.client = client
        self.model = model
        self.replay_reasoning = replay_reasoning

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: tuple[ToolSpec, ...] = (),
        settings: ModelSettings,
        kind: Literal["agent", "compaction"] = "agent",
    ) -> ModelCall:
        wire = messages_to_wire(messages, replay_reasoning=self.replay_reasoning)
        options = settings.model_dump(exclude_none=True)
        if options.get("parallel_tool_calls") is True and not tools:
            options.pop("parallel_tool_calls")
        if "top_logprobs" in options:
            options["logprobs"] = True
        extra = options["extra_body"]
        if set(extra) & {"messages", "model", "tools", "stream", "tool_choice"}:
            raise ValueError("extra_body cannot override the conversation or execution contract")
        request = {"model": self.model, "messages": wire, **options}
        if tools:
            request["tools"] = [tool.to_wire() for tool in tools]
        response = await self.client.chat.completions.create(**request)
        if not response.choices:
            provider_error = getattr(response, "error", None)
            details = f": {provider_error}" if provider_error is not None else ""
            raise RuntimeError(
                f"ChatCompletion response has no choices (possible provider error payload){details}"
            )
        choice = response.choices[0]
        message = choice.message
        raw = message.model_dump(mode="json", exclude_unset=True)
        if raw.get("thinking_blocks"):
            raise ValueError("Signed thinking-block replay is not currently supported")
        if raw.get("audio"):
            raise ValueError("Audio is not currently supported")
        calls = []
        for call in message.tool_calls or ():
            if call.type != "function":
                raise ModelBehaviorError(f"Unsupported tool call type: {call.type}")
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
        assistant = AssistantMessage(
            content=message.content,
            reasoning_content=raw.get("reasoning_content") or raw.get("reasoning"),
            refusal=message.refusal,
            tool_calls=tuple(calls),
        )
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return ModelCall(
            assistant=assistant,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
        )


def messages_to_wire(messages: Sequence[Message], *, replay_reasoning: bool = True) -> list[dict]:
    output = []
    images = []
    for message in messages:
        # All parallel results must precede the image-bearing user message.
        if images and not isinstance(message, ToolMessage):
            output.append({"role": "user", "content": images})
            images = []
        if isinstance(message, SystemMessage):
            output.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            output.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage):
            content = message.content
            if isinstance(content, list) and any(part["type"] == "image_url" for part in content):
                images.append({"type": "text", "text": f"Images returned by tool call {message.tool_call_id}:"})
                images.extend(part for part in content if part["type"] == "image_url")
                content = [part for part in content if part["type"] == "text"]
                content.append({"type": "text", "text": "Tool images are attached in the following user message."})
            output.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": content,
                }
            )
        elif isinstance(message, AssistantMessage):
            if not (message.content or message.tool_calls or message.refusal):
                continue  # Reference converter does not replay standalone reasoning/empty messages.
            wire = {"role": "assistant"}
            if message.content:
                wire["content"] = message.content
            if message.tool_calls:
                if not message.refusal:
                    wire.setdefault("content", None)
                wire["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments or "{}",
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.refusal:
                wire["refusal"] = message.refusal
            if replay_reasoning and message.reasoning_content:
                wire["reasoning_content"] = message.reasoning_content
            output.append(wire)
        else:
            raise TypeError(f"Unsupported message: {type(message).__name__}")
    if images:
        output.append({"role": "user", "content": images})
    return output
