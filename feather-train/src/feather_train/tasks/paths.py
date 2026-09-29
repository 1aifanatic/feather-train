from __future__ import annotations


def is_safe_task_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and not any(character in value for character in "*?[]")
        and value.isprintable()
    )
