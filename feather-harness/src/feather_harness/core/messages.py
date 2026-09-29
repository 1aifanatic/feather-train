"""Feather's cumulative conversation format, independent of any agent SDK."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(FrozenModel):
    id: str
    name: str
    arguments: str


class SystemMessage(FrozenModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(FrozenModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(FrozenModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: Annotated[str, Field(pattern=r"^data:image/(png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+$")]


class ImageBlock(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


ToolContent = str | list[TextBlock | ImageBlock]


class ToolMessage(FrozenModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: ToolContent


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]
MESSAGES = TypeAdapter(tuple[Message, ...])


def transcript_jsonl(messages: tuple[Message, ...] | list[Message]) -> str:
    return "".join(message.model_dump_json() + "\n" for message in messages)
