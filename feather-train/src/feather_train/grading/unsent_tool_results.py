"""Count tool bodies left unread when Feather exhausts its model turns."""

from feather_harness.core.messages import ToolMessage
from feather_harness.core.model import messages_to_wire
from feather_harness.core.runner import RunResult
from renderers import RenderedTokens


def get_unsent_tool_turn(result: RunResult) -> list[dict]:
    """Return the last assistant call and its unread results after a turn-limit stop."""
    if result.stop_reason != "MaxTurnsExceeded":
        return []
    messages = result.segments[-1].messages
    start = len(messages)
    while start and isinstance(messages[start - 1], ToolMessage):
        start -= 1
    if start == len(messages):
        return []
    # The preceding assistant supplies tool names/IDs required by some templates.
    return messages_to_wire(messages[start - 1 :])


def count_tool_content_tokens(rendered: RenderedTokens) -> int | None:
    if (
        len(rendered.is_content) != len(rendered.token_ids)
        or len(rendered.message_indices) != len(rendered.token_ids)
        or not rendered.message_roles
    ):
        return None
    return sum(
        end - start
        for start, end in rendered.content_token_spans_by_role().get("tool", [])
    )
