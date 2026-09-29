from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from feather_harness import RunResult
from pydantic import BaseModel

AGENT_TRACE_SCHEMA_VERSION = "tldc.feather-agent-trace.v2"


class Criterion(BaseModel):
    name: str
    met: bool
    weight: int | float = 1
    explanation: str | None = None
    metric: str | None = None
    criterion_resolution_state: Literal["resolved", "unresolved", "ambiguous"]
    reported_criterion_id: str | None = None
    reported_section_id: str | None = None
    criterion_id: str | None = None
    section_id: str | None = None
    section_name: str | None = None
    section_category: str | None = None
    rubric_category: str | None = None
    verdict: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class AgentEvidence:
    """Portable, JSON-safe evidence emitted by one Feather run."""

    model: str
    instructions: str | None
    trace: RunResult

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": AGENT_TRACE_SCHEMA_VERSION,
            "agent": {"name": "feather", "model": self.model},
            "instructions": self.instructions,
            "trace": self.trace.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class JudgeAttemptFailure:
    attempt: int
    message: str
    retriable: bool = True
    exit_code: int | None = None

    def as_error(self) -> dict[str, object]:
        error: dict[str, object] = {
            "attempt": self.attempt,
            "message": self.message,
            "retriable": self.retriable,
        }
        if self.exit_code is not None:
            error["exit_code"] = self.exit_code
        return error


@dataclass(slots=True)
class GradeStep:
    name: str
    start: float
    end: float
    ok: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GradeResult:
    score: float
    num_met: int | None
    num_criteria: int | None
    criteria: tuple[dict, ...]
    stdout: str
    failed_attempts: list[JudgeAttemptFailure]
    steps: tuple[GradeStep, ...] = ()
    judge_usage: dict[str, object] = field(default_factory=dict)
    report_error: str | None = None
