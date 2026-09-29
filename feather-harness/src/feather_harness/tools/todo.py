"""The in-memory todo list, its tool, and compaction state."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, ValidationError

from ..core.messages import FrozenModel
from . import Tool, ToolResult, ToolSpec, invalid_arguments_result, truncate_tool_output

TODO_WRITE_DESCRIPTION = (
    "Read or replace the todo list for the current session. Omit `todos` to return the current "
    "list; provide `todos` to replace the full list, or an empty list to clear it."
)

MAX_TODO_ITEMS = 20
MAX_TODO_CONTENT_CHARS = 160


class TodoItem(FrozenModel):
    content: str = Field(
        min_length=1,
        max_length=MAX_TODO_CONTENT_CHARS,
        description="Text of the todo item.",
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Current status of the todo item."
    )


class TodoWriteArgs(FrozenModel):
    todos: list[TodoItem] = Field(
        default_factory=list,
        max_length=MAX_TODO_ITEMS,
        description=(
            "The complete replacement todo list. Omit to read the current list without "
            "changing it; pass an empty list to clear it."
        ),
    )


class TodoList:
    def __init__(self) -> None:
        self._todos: tuple[TodoItem, ...] | None = None

    def tool(self) -> Tool:
        return Tool(
            spec=ToolSpec(
                name="todo_write",
                description=TODO_WRITE_DESCRIPTION,
                parameters=TodoWriteArgs.model_json_schema(),
            ),
            invoke=truncate_tool_output(self._invoke),
        )

    def compaction_state(self) -> tuple[str, ...]:
        if self._todos is None:
            return ()
        return (f"## Current Todo List\n\n{self._render()}",)

    async def _invoke(self, raw_arguments: str) -> ToolResult:
        try:
            args = TodoWriteArgs.model_validate_json(raw_arguments)
        except ValidationError as exc:
            return invalid_arguments_result(exc)
        if "todos" not in args.model_fields_set:
            return ToolResult(content=self._render())
        self._todos = tuple(args.todos)
        return ToolResult(content="Todo list updated.")

    def _render(self) -> str:
        return json.dumps(
            {"todos": [todo.model_dump(mode="json") for todo in self._todos or ()]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
