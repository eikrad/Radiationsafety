"""Tests for the generation grader chain and grade_generation node."""

from graph.chains.generation_grader import GradeGeneration, get_generation_grader


# ---------------------------------------------------------------------------
# GradeGeneration schema
# ---------------------------------------------------------------------------


def test_grade_generation_passed_true_has_empty_missing_info():
    """When passed=True, missing_info must be empty (grader found nothing missing)."""
    score = GradeGeneration(passed=True, missing_info="")
    assert score.passed is True
    assert score.missing_info == ""


def test_grade_generation_passed_false_carries_missing_info():
    """When passed=False, missing_info describes what was missing."""
    score = GradeGeneration(passed=False, missing_info="dose limits table Annex 2 GSR-3")
    assert score.passed is False
    assert score.missing_info == "dose limits table Annex 2 GSR-3"


def test_grade_generation_missing_info_defaults_to_empty():
    """missing_info has a default of empty string so callers need not supply it."""
    score = GradeGeneration(passed=True)
    assert score.missing_info == ""
