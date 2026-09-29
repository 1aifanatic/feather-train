"""Reconcile grader reports with authored rubric identities for logging."""

from feather_train.grading.types import Criterion, GradeResult


def build_judge_span_output(result: GradeResult) -> dict[str, object]:
    output = {
        "score": result.score,
        "num_met": result.num_met,
        "num_criteria": result.num_criteria,
        "stdout": result.stdout[-1500:],
        "criteria": list(result.criteria),
        "errors": [failure.as_error() for failure in result.failed_attempts],
        "judge_usage": result.judge_usage,
    }
    if result.report_error is not None:
        output["report_error"] = result.report_error
    return output


def report_to_criteria(
    report: list[dict],
    *,
    rubric: list[dict],
) -> list[dict]:
    """Convert the harbor grader's rubric.EvaluationReport ``report`` entries
    (``CriterionReport``: requirement/verdict/weight/reason) into shared
    ``Criterion`` dicts (name/met/weight/explanation) for the eval uploader + UI.
    """
    out: list[dict] = []
    for entry in report:
        if not isinstance(entry, dict):
            continue
        identity = _resolve_criterion_identity(entry, rubric)
        out.append(
            Criterion(
                name=entry.get("requirement", ""),
                met=entry.get("verdict") == "MET",
                weight=float(entry.get("weight", 1.0)),
                metric="harbor_grader_reward",
                explanation=entry.get("reason", ""),
                **identity,
                verdict=(str(entry["verdict"]) if entry.get("verdict") is not None else None),
                score=(float(entry["score"]) if entry.get("score") is not None else None),
            ).model_dump(exclude_none=True)
        )
    return out


def _resolve_criterion_identity(
    report: dict,
    identities: list[dict[str, str | None]],
) -> dict[str, str | None]:
    reported_criterion_id = report.get("criterion_id") or report.get("id")
    reported_section_id = report.get("section_id")
    explicit_id = str(reported_criterion_id) if reported_criterion_id is not None else None
    explicit_section = str(reported_section_id) if reported_section_id is not None else None
    evidence = {
        "reported_criterion_id": explicit_id,
        "reported_section_id": explicit_section,
    }
    candidates = identities
    if explicit_section:
        candidates = [item for item in candidates if item.get("section_id") == explicit_section]
    if explicit_id is not None:
        matches = [item for item in candidates if item.get("criterion_id") == explicit_id]
    else:
        requirement = _normalized_requirement(report.get("requirement"))
        matches = [
            item
            for item in candidates
            if requirement is not None
            and _normalized_requirement(item.get("criteria")) == requirement
        ]
    return _criterion_resolution(matches, evidence=evidence)


def _criterion_resolution(
    matches: list[dict[str, str | None]],
    *,
    evidence: dict[str, str | None],
) -> dict[str, str | None]:
    if len(matches) == 1 and matches[0].get("section_id") and matches[0].get("criterion_id"):
        match = matches[0]
        return {
            **evidence,
            "criterion_id": match.get("criterion_id"),
            "section_id": match.get("section_id"),
            "section_name": match.get("section_name"),
            "section_category": match.get("section_category"),
            "rubric_category": match.get("rubric_category"),
            "criterion_resolution_state": "resolved",
        }
    return {
        **evidence,
        "criterion_resolution_state": "ambiguous" if len(matches) > 1 else "unresolved",
    }


def _normalized_requirement(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.casefold().split())
    return normalized or None
