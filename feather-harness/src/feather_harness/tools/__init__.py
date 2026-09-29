"""The loop needs a model-facing schema and one raw-arguments invocation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, ValidationError

from ..core.messages import FrozenModel, ToolContent
from ..scripts._files import MAX_OUTPUT_CHARS, truncate_output


class ToolSpec(FrozenModel):
    name: str
    description: str
    parameters: dict[str, JsonValue]
    strict: bool = False

    def to_wire(self) -> dict:
        return {"type": "function", "function": self.model_dump(mode="json")}


class ToolResult(FrozenModel):
    """Transport envelope; only content enters the model conversation."""

    content: ToolContent
    error: Literal["invalid_arguments", "invalid_path", "tool_error"] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    invoke: Callable[[str], Awaitable[ToolResult]]


def truncate_tool_output(
    invoke: Callable[[str], Awaitable[ToolResult]],
) -> Callable[[str], Awaitable[ToolResult]]:
    """Cap todo/web text responses without saving a file."""

    async def invoke_bounded(raw_arguments: str) -> ToolResult:
        result = await invoke(raw_arguments)
        if not isinstance(result.content, str) or len(result.content) <= MAX_OUTPUT_CHARS:
            return result
        return result.model_copy(update={"content": truncate_output(result.content)})

    return invoke_bounded


def invalid_arguments_result(error: ValidationError) -> ToolResult:
    lines = []
    for detail in error.errors(include_url=False, include_input=False):
        if detail["type"] == "json_invalid":
            lines.append(f"- Arguments must be valid JSON: {detail['ctx']['error']}.")
            lines.append(
                "- Send exactly one JSON object matching the schema, with no text or extra braces before or after it."
            )
        else:
            location = ".".join(str(part) for part in detail["loc"]) or "arguments"
            lines.append(f"- {location}: {detail['msg']}.")
    return ToolResult(
        content="Invalid tool arguments:\n"
        + "\n".join(lines)
        + "\nRe-send the call with arguments that match the tool's schema.",
        error="invalid_arguments",
    )
