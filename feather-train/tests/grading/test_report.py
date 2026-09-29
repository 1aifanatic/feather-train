import pytest

from feather_train.grading.report import report_to_criteria


def test_criterion_identity_does_not_guess_between_duplicate_requirements():
    rubric = [
        {"criterion_id": "one", "section_id": "section", "criteria": "Same"},
        {"criterion_id": "two", "section_id": "section", "criteria": "Same"},
    ]
    results = report_to_criteria(
        [
            {"requirement": "same", "verdict": "MET"},
            {"criterion_id": "two", "requirement": "Same", "verdict": "UNMET"},
        ],
        rubric=rubric,
    )
    assert results[0]["criterion_resolution_state"] == "ambiguous"
    assert "criterion_id" not in results[0]
    assert results[1]["criterion_id"] == "two"
    assert results[1]["met"] is False


@pytest.mark.parametrize(
    ("reported", "state", "criterion_id", "section_id"),
    [
        ({"criterion_id": "unknown", "requirement": "Shared"}, "unresolved", None, None),
        ({"criterion_id": "one", "section_id": "second"}, "unresolved", None, None),
        ({"requirement": "  SHARED \n", "section_id": "second"}, "resolved", "two", "second"),
        ({"criterion_id": "repeated"}, "ambiguous", None, None),
        ({"criterion_id": "repeated", "section_id": "second"}, "resolved", "repeated", "second"),
        ({"criterion_id": "orphan"}, "unresolved", None, None),
        ({"requirement": "  "}, "unresolved", None, None),
    ],
    ids=[
        "unknown-id-no-text-fallback",
        "wrong-section",
        "normalized-text-in-section",
        "duplicate-id",
        "id-in-section",
        "missing-section-id",
        "empty-requirement",
    ],
)
def test_report_identity_requires_an_unambiguous_authored_match(
    reported, state, criterion_id, section_id
):
    rubric = [
        {"criterion_id": "one", "section_id": "first", "criteria": "Shared"},
        {"criterion_id": "repeated", "section_id": "first", "criteria": "First"},
        {"criterion_id": "two", "section_id": "second", "criteria": "Shared"},
        {"criterion_id": "repeated", "section_id": "second", "criteria": "Second"},
        {"criterion_id": "orphan", "section_id": None, "criteria": "Orphan"},
    ]
    (result,) = report_to_criteria([reported], rubric=rubric)
    assert result["criterion_resolution_state"] == state
    assert result.get("criterion_id") == criterion_id
    assert result.get("section_id") == section_id
    assert result.get("reported_criterion_id") == reported.get("criterion_id")
    assert result.get("reported_section_id") == reported.get("section_id")
