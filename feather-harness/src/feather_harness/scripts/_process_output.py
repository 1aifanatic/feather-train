"""Output limits and model feedback for managed Bash processes."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from process_worker import Task

MAX_LOG_BYTES = 1_000_000
MAX_PREVIEW_CHARS = 23_000
MAX_TASK_LABEL_CHARS = 160
_LOG_MARKER_RESERVE = 128


@dataclass
class _BoundedOutput:
    total_bytes: int = 0
    _prefix: bytes = b""
    _suffix: bytes = b""

    @property
    def truncated(self) -> bool:
        return self.total_bytes > MAX_LOG_BYTES

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        previous_total = self.total_bytes
        self.total_bytes += len(chunk)
        if self.total_bytes <= MAX_LOG_BYTES:
            self._prefix += chunk
            return

        retained_limit = MAX_LOG_BYTES - _LOG_MARKER_RESERVE
        prefix_limit = retained_limit // 2
        suffix_limit = retained_limit - prefix_limit
        # On the first overflow, preserve both ends before discarding the middle.
        if previous_total <= MAX_LOG_BYTES:
            combined = self._prefix + chunk
            self._prefix = combined[:prefix_limit]
            self._suffix = combined[-suffix_limit:]
            return
        self._suffix = (self._suffix + chunk)[-suffix_limit:]

    def bytes(self) -> bytes:
        if not self.truncated:
            return self._prefix
        omitted = self.total_bytes - len(self._prefix) - len(self._suffix)
        marker = f"\n\n[... {omitted} output bytes omitted ...]\n\n".encode()
        return self._prefix + marker + self._suffix

    def preview(self) -> tuple[str, bool]:
        text = self.bytes().decode("utf-8", errors="replace")
        if len(text) <= MAX_PREVIEW_CHARS:
            return text, self.truncated
        marker = "\n\n[... output characters omitted ...]\n\n"
        content_limit = MAX_PREVIEW_CHARS - len(marker)
        head = content_limit // 2
        tail = content_limit - head
        return f"{text[:head]}{marker}{text[-tail:]}", True


def format_process_result(task: "Task") -> str:
    preview, truncated = task.output.preview()
    sections = []
    if task.task_id != "foreground":
        sections.append(f"Background task: {task.task_id}")
    sections.append(f"State: {task.state}")
    if task.exit_code is not None:
        sections.append(f"Exit code: {task.exit_code}")
    if truncated:
        sections.append(
            f"Output truncated: showing a bounded preview of {task.output.total_bytes} bytes."
        )
    if task.log_path is not None:
        sections.append(f"Log: {task.log_path}")
    sections.extend(("Output:", preview))
    return "\n".join(sections)


def format_process_list(
    tasks: Sequence["Task"],
    *,
    active_only: bool,
    total: int,
) -> str:
    heading = "Active background tasks" if active_only else "Background tasks"
    if not tasks:
        return f"{heading}: none"
    if len(tasks) < total:
        lines = [f"Showing {len(tasks)} of {total} {heading.lower()}:"]
    else:
        lines = [f"{heading}: {total}"]
    lines.extend(format_task_list_entry(task) for task in tasks)
    return "\n".join(lines)


def format_task_list_entry(task: "Task") -> str:
    state = task.state
    if task.state == "failed" and not task.monitor.done():
        state = f"{state}, cleanup pending"
    if task.exit_code is not None:
        state = f"{state}, exit {task.exit_code}"
    return f"- {task.task_id} [{state}] {_task_label(task)}"


def _task_label(task: "Task") -> str:
    label = " ".join((task.description or "").split())
    if not label:
        label = " ".join(task.command.split())
    if len(label) <= MAX_TASK_LABEL_CHARS:
        return label
    return f"{label[: MAX_TASK_LABEL_CHARS - 3]}..."
