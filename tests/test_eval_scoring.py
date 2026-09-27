"""Tests for deterministic scoring: evidence matching, recall, error attribution."""

import pytest
from langchain_core.documents import Document

from eval.scoring import evidence_found, score_item

ANNEX_2 = "Dosisgrænser for erhvervsmæssig bestråling fremgår af bilag 2."
PER_YEAR = "Dosisgrænserne gælder for et kalenderår."

ITEM = {
    "id": "dk-dose-limits",
    "expected_behavior": "answer",
    "nuggets": [
        {
            "text": "Dose limits are in Annex 2",
            "importance": "vital",
            "evidence": ["fremgår af bilag 2", "anført i bilag 2"],
        },
        {
            "text": "The limit is 20 mSv per year",
            "importance": "vital",
            "evidence": ["20 mSv pr. år"],
        },
        {
            "text": "Limits apply per calendar year",
            "importance": "okay",
            "evidence": ["gælder for et kalenderår"],
        },
    ],
}

REFUSAL = {"id": "out-of-scope", "expected_behavior": "refuse", "nuggets": []}

ALL_EVIDENCE = [
    Document(page_content=ANNEX_2),
    Document(page_content="Grænsen er 20 mSv pr. år."),
    Document(page_content=PER_YEAR),
]


def _output(initial=None, context_docs=None, sufficient=True):
    initial = ALL_EVIDENCE if initial is None else initial
    context_docs = initial if context_docs is None else context_docs
    return {
        "initial_documents": initial,
        "context": "\n\n".join(d.page_content for d in context_docs),
        "sufficient": sufficient,
    }


def _verdict(labels=("support", "support", "support"), unsupported=(), refused=False):
    return {
        "nuggets": list(labels),
        "unsupported_claims": list(unsupported),
        "refused": refused,
    }


# --- evidence matching -------------------------------------------------------


def test_evidence_matches_despite_case_and_line_breaks():
    chunk = "Dosisgrænser for erhvervsmæssig\n  bestråling FREMGÅR af Bilag 2."

    assert evidence_found(["fremgår af bilag 2"], [chunk])


def test_evidence_matches_despite_soft_hyphens_and_typographic_dashes():
    chunk = "indsats\u2011 og redningsmand\u00adskab \u2013 se bilag 3"

    assert evidence_found(["indsats- og redningsmandskab - se bilag 3"], [chunk])


def test_evidence_matches_despite_invisible_format_characters():
    """The Danish XML text contains zero-width joiners inside words."""
    chunk = "resultatet af individuel dosi\u200dsovervågning"

    assert evidence_found(["individuel dosisovervågning"], [chunk])


def test_any_alternative_quote_counts_as_evidence():
    assert evidence_found(["not there", "bilag 2"], ["… anført i bilag 2 …"])


def test_a_paraphrase_is_not_evidence():
    assert not evidence_found(["fremgår af bilag 2"], ["står i anneks to"])


# --- answerable questions ----------------------------------------------------


def test_a_fully_supported_grounded_answer_passes():
    scores = score_item(ITEM, _output(), _verdict())

    assert scores["pass"] is True
    assert scores["error_type"] == "ok"
    assert scores["vital_recall"] == 1.0
    assert scores["evidence_recall_initial"] == 1.0
    assert scores["evidence_recall_context"] == 1.0


def test_a_missing_vital_fact_with_its_evidence_in_context_is_a_generator_miss():
    scores = score_item(
        ITEM, _output(), _verdict(("support", "not_support", "support"))
    )

    assert scores["pass"] is False
    assert scores["error_type"] == "generator_miss"
    assert scores["vital_recall"] == 0.5
    assert scores["context_utilization"] == 0.5


def test_a_missing_vital_fact_whose_evidence_was_never_retrieved_is_a_retrieval_miss():
    retrieved = [Document(page_content=ANNEX_2)]
    scores = score_item(
        ITEM, _output(retrieved), _verdict(("support", "not_support", "support"))
    )

    assert scores["error_type"] == "retrieval_miss"
    assert scores["evidence_recall_initial"] == 0.5
    assert scores["evidence_recall_context"] == 0.5


def test_evidence_found_later_by_retrieve_missing_counts_for_the_context_only():
    first = [Document(page_content=ANNEX_2)]
    scores = score_item(ITEM, _output(first, ALL_EVIDENCE), _verdict())

    assert scores["evidence_recall_initial"] == 0.5
    assert scores["evidence_recall_context"] == 1.0
    assert scores["pass"] is True


def test_partial_support_counts_half_only_in_the_lenient_recall():
    scores = score_item(
        ITEM, _output(), _verdict(("support", "partial_support", "support"))
    )

    assert scores["vital_recall"] == 0.5
    assert scores["vital_recall_lenient"] == 0.75
    assert scores["pass"] is False


def test_okay_nuggets_count_in_all_recall_but_not_for_passing():
    scores = score_item(
        ITEM, _output(), _verdict(("support", "support", "not_support"))
    )

    assert scores["pass"] is True
    assert scores["all_recall"] == pytest.approx(2 / 3)


def test_an_unsupported_claim_fails_an_otherwise_complete_answer():
    scores = score_item(ITEM, _output(), _verdict(unsupported=["The limit is 50 mSv"]))

    assert scores["pass"] is False
    assert scores["error_type"] == "unsupported_claim"
    assert scores["unsupported_claims"] == 1


def test_an_answer_from_model_knowledge_is_not_grounded():
    """Vital facts stated without their evidence in context: correct maybe, but not grounded."""
    no_evidence = [Document(page_content="Unrelated chunk about transport.")]
    scores = score_item(ITEM, _output(no_evidence), _verdict())

    assert scores["vital_recall"] == 1.0
    assert scores["grounded_vital_recall"] == 0.0
    assert scores["context_utilization"] is None
    assert scores["pass"] is False
    assert scores["error_type"] == "retrieval_miss"


def test_grounded_recall_counts_only_facts_whose_evidence_was_in_context():
    retrieved = [Document(page_content=ANNEX_2)]
    scores = score_item(
        ITEM, _output(retrieved), _verdict(("support", "support", "support"))
    )

    assert scores["grounded_vital_recall"] == 0.5


def test_refusing_although_the_evidence_was_there_is_a_wrong_refusal():
    scores = score_item(ITEM, _output(), _verdict(("not_support",) * 3, refused=True))

    assert scores["error_type"] == "wrong_refusal"
    assert scores["refused"] is True


def test_refusing_when_the_evidence_was_not_retrieved_is_blamed_on_retrieval():
    scores = score_item(
        ITEM,
        _output([Document(page_content="nothing relevant")]),
        _verdict(("not_support",) * 3, refused=True),
    )

    assert scores["error_type"] == "retrieval_miss"


# --- refusal questions -------------------------------------------------------


def test_refusing_an_out_of_scope_question_passes():
    scores = score_item(REFUSAL, _output(), _verdict((), refused=True))

    assert scores["pass"] is True
    assert scores["error_type"] == "ok"
    assert scores["vital_recall"] is None
    assert scores["evidence_recall_initial"] is None


def test_answering_an_out_of_scope_question_is_a_missed_refusal():
    scores = score_item(REFUSAL, _output(), _verdict((), refused=False))

    assert scores["pass"] is False
    assert scores["error_type"] == "missed_refusal"


# --- grade_documents accuracy ------------------------------------------------


@pytest.mark.parametrize(
    ("item", "initial", "sufficient", "correct"),
    [
        (ITEM, ALL_EVIDENCE, True, True),
        (ITEM, ALL_EVIDENCE, False, False),
        (ITEM, [Document(page_content=ANNEX_2)], True, False),
        (ITEM, [Document(page_content=ANNEX_2)], False, True),
        (REFUSAL, ALL_EVIDENCE, False, True),
        (REFUSAL, ALL_EVIDENCE, True, False),
    ],
)
def test_the_sufficiency_grader_is_right_when_it_matches_the_evidence(
    item, initial, sufficient, correct
):
    labels = () if item is REFUSAL else ("support",) * 3
    verdict = _verdict(labels, refused=item is REFUSAL)
    scores = score_item(item, _output(initial, sufficient=sufficient), verdict)

    assert scores["grade_documents_correct"] is correct


def test_no_sufficiency_verdict_means_no_grader_accuracy():
    scores = score_item(ITEM, _output(sufficient=None), _verdict())

    assert scores["grade_documents_correct"] is None


# --- judge failures ----------------------------------------------------------


def test_a_failed_judge_leaves_the_question_unscored_but_keeps_evidence_metrics():
    scores = score_item(ITEM, _output(), None)

    assert scores["error_type"] == "judge_error"
    assert scores["pass"] is None
    assert scores["vital_recall"] is None
    assert scores["evidence_recall_context"] == 1.0


def test_a_verdict_with_the_wrong_number_of_nugget_labels_is_rejected():
    with pytest.raises(ValueError, match="3 nuggets but 2 labels"):
        score_item(ITEM, _output(), _verdict(("support", "support")))
