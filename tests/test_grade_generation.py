"""Tests for the generation grader chain and grade_generation node."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from graph.chains.generation_grader import GradeGeneration, PartVerdict
from graph.state import GraphState

# ---------------------------------------------------------------------------
# GradeGeneration: the verdict follows from the part-by-part assessment
# ---------------------------------------------------------------------------


FACTS = (
    "[Source: GSR Part 3 (IAEA)] The effective dose limit for occupational exposure "
    "is 20 mSv per year, averaged over five consecutive years."
)
QUOTE = "20 mSv per year, averaged over five consecutive years"


def _parts(*statuses: str, quote: str = QUOTE) -> list[PartVerdict]:
    return [
        PartVerdict(part=f"part {n}", status=s, quote=quote)
        for n, s in enumerate(statuses, 1)
    ]


def _verdict(passed: bool, missing_info: str = "") -> GradeGeneration:
    """A grader reply whose parts make it pass or fail against FACTS."""
    status = "answered_supported" if passed else "not_answered"
    return GradeGeneration(parts=_parts(status), missing_info=missing_info)


def test_an_answer_passes_when_every_part_is_answered_from_the_facts():
    assert GradeGeneration(
        parts=_parts("answered_supported", "answered_supported")
    ).passes(FACTS)


def test_an_answer_that_leaves_one_part_open_fails():
    """The grader once listed every part as not answered and still said passed (cs137)."""
    assert not GradeGeneration(
        parts=_parts("answered_supported", "not_answered")
    ).passes(FACTS)


def test_an_answer_with_a_part_the_facts_do_not_state_fails():
    assert not GradeGeneration(
        parts=_parts("answered_supported", "answered_not_in_facts")
    ).passes(FACTS)


def test_a_refusal_passes_only_when_the_facts_are_off_topic_for_every_part():
    assert GradeGeneration(parts=_parts("facts_off_topic", "facts_off_topic")).passes(
        FACTS
    )
    assert not GradeGeneration(
        parts=_parts("facts_off_topic", "answered_supported")
    ).passes(FACTS)


def test_a_reply_that_lists_no_parts_does_not_pass():
    assert not GradeGeneration(parts=[]).passes(FACTS)


def test_a_supported_part_needs_a_quote_the_facts_contain():
    """The grader took the answer's reading of the context for 7 of 7 generator
    misses; quoting the Facts makes it look there itself (ROADMAP, Step G1)."""
    assert not GradeGeneration(parts=_parts("answered_supported", quote="")).passes(
        FACTS
    )
    assert not GradeGeneration(
        parts=_parts("answered_supported", quote="10 mSv per year")
    ).passes(FACTS)


def test_a_quote_matches_despite_case_spacing_quotes_and_elisions():
    quote = '"The effective dose limit ... is 20  mSv per year, averaged"'
    assert GradeGeneration(parts=_parts("answered_supported", quote=quote)).passes(
        FACTS
    )


def test_a_refusal_needs_no_quote():
    assert GradeGeneration(parts=_parts("facts_off_topic", quote="")).passes(FACTS)


def test_missing_info_defaults_to_empty():
    assert _verdict(True).missing_info == ""


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
        "documents": documents or [Document(page_content=FACTS)],
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

    with _patch_grader(_verdict(True)):
        out = grade_generation(_make_state())

    assert out["reflection"] == ""


def test_grade_generation_node_sets_missing_info_as_reflection_on_fail():
    """When grader fails with a hint, reflection carries that hint."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(_verdict(False, "Annex 2 dose table")):
        out = grade_generation(_make_state())

    assert out["reflection"] == "Annex 2 dose table"


def test_grade_generation_node_uses_sentinel_when_missing_info_empty_on_fail():
    """When grader fails but provides no hint, reflection is set to sentinel 'retry'."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(_verdict(False)):
        out = grade_generation(_make_state())

    assert out["reflection"] == "retry"


def test_grade_generation_node_fails_an_answer_whose_quoted_support_is_not_in_the_context():
    from graph.nodes.grade_generation import grade_generation

    grade = GradeGeneration(
        parts=_parts("answered_supported", quote="10 mSv per year"),
        missing_info="occupational dose limit GSR Part 3",
    )
    with _patch_grader(grade):
        out = grade_generation(_make_state(context_used=FACTS))

    assert out["generation_passed_grading"] is False
    assert out["reflection"] == "occupational dose limit GSR Part 3"


def test_grade_generation_node_sets_generation_passed_grading_flag():
    """Node writes generation_passed_grading=True when passed, False when not."""
    from graph.nodes.grade_generation import grade_generation

    with _patch_grader(_verdict(True)):
        out_pass = grade_generation(_make_state())
    with _patch_grader(_verdict(False, "something")):
        out_fail = grade_generation(_make_state())

    assert out_pass["generation_passed_grading"] is True
    assert out_fail["generation_passed_grading"] is False
