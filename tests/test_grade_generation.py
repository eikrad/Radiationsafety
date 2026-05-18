"""Tests for the generation grader chain and grade_generation node."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from graph.chains.generation_grader import GradeGeneration, get_generation_grader
from graph.state import GraphState


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


# ---------------------------------------------------------------------------
# grade_generation node
# ---------------------------------------------------------------------------


def _make_state(
    *,
    generation: str = "Some answer.",
    documents: list | None = None,
    context_used: str = "",
    web_search_attempted: bool = False,
    retry_count: int = 0,
) -> GraphState:
    return {
        "question": "What are the dose limits?",
        "generation": generation,
        "documents": documents or [Document(page_content="GSR-3 dose limits...")],
        "context_used_for_generation": context_used,
        "web_search": False,
        "web_search_attempted": web_search_attempted,
        "retry_after_generation_count": retry_count,
    }


def _patch_grader(grade: GradeGeneration):
    """Return a context manager that patches get_generation_grader in the node module."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = grade
    return patch(
        "graph.nodes.grade_generation.get_generation_grader",
        return_value=mock_chain,
    )


def test_grade_generation_node_sets_empty_reflection_on_pass():
    """When grader passes, reflection is reset to empty string."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(GradeGeneration(passed=True)):
        out = grade_generation(_make_state())

    assert out["reflection"] == ""


def test_grade_generation_node_sets_missing_info_as_reflection_on_fail():
    """When grader fails with a hint, reflection carries that hint."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(GradeGeneration(passed=False, missing_info="Annex 2 dose table")):
        out = grade_generation(_make_state())

    assert out["reflection"] == "Annex 2 dose table"


def test_grade_generation_node_uses_sentinel_when_missing_info_empty_on_fail():
    """When grader fails but provides no hint, reflection is set to sentinel 'retry'."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(GradeGeneration(passed=False, missing_info="")):
        out = grade_generation(_make_state())

    assert out["reflection"] == "retry"


def test_grade_generation_node_sets_generation_passed_grading_flag():
    """Node writes generation_passed_grading=True when passed, False when not."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(GradeGeneration(passed=True)):
        out_pass = grade_generation(_make_state())
    with _patch_grader(GradeGeneration(passed=False, missing_info="something")):
        out_fail = grade_generation(_make_state())

    assert out_pass["generation_passed_grading"] is True
    assert out_fail["generation_passed_grading"] is False
