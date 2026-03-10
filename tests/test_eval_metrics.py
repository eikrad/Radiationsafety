"""Tests for eval metrics (mocked graders to avoid LLM calls)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from eval.metrics import (
    answer_relevance,
    compute_all_metrics,
    context_precision,
    context_recall,
    faithfulness,
)


def _mock_grade_gen(grounded: bool, answers_question: bool):
    return MagicMock(grounded=grounded, answers_question=answers_question)


def _mock_grade_suff(binary_score: bool):
    return MagicMock(binary_score=binary_score)


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Dose limits for workers are defined in the regulations."),
    ]


def test_faithfulness_returns_1_when_grounded(sample_docs):
    with patch("eval.metrics.get_generation_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_gen(grounded=True, answers_question=True)
        m.return_value = grader
        assert faithfulness("Q?", "Answer.", sample_docs) == 1.0


def test_faithfulness_returns_0_when_not_grounded(sample_docs):
    with patch("eval.metrics.get_generation_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_gen(grounded=False, answers_question=True)
        m.return_value = grader
        assert faithfulness("Q?", "Answer.", sample_docs) == 0.0


def test_faithfulness_returns_0_for_empty_generation(sample_docs):
    assert faithfulness("Q?", "", sample_docs) == 0.0


def test_answer_relevance_returns_1_when_answers(sample_docs):
    with patch("eval.metrics.get_generation_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_gen(grounded=True, answers_question=True)
        m.return_value = grader
        assert answer_relevance("Q?", "Answer.", sample_docs) == 1.0


def test_answer_relevance_returns_0_when_not_answers(sample_docs):
    with patch("eval.metrics.get_generation_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_gen(grounded=True, answers_question=False)
        m.return_value = grader
        assert answer_relevance("Q?", "Answer.", sample_docs) == 0.0


def test_context_precision_returns_1_when_sufficient(sample_docs):
    with patch("eval.metrics.get_context_sufficiency_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_suff(True)
        m.return_value = grader
        assert context_precision("Q?", sample_docs) == 1.0


def test_context_precision_returns_0_when_not_sufficient(sample_docs):
    with patch("eval.metrics.get_context_sufficiency_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_suff(False)
        m.return_value = grader
        assert context_precision("Q?", sample_docs) == 0.0


def test_context_precision_returns_0_for_no_documents():
    assert context_precision("Q?", []) == 0.0


def test_context_recall_with_key_facts():
    docs = [Document(page_content="We have record-keeping and retention rules.")]
    # "record-keeping" and "retention" in context, "xylophone" not
    score = context_recall("Q?", docs, key_facts=["record-keeping", "retention", "xylophone"])
    assert 0 < score < 1.0
    assert abs(score - 2 / 3) < 0.01


def test_context_recall_with_key_facts_all_found():
    docs = [Document(page_content="record-keeping and retention for environmental monitoring.")]
    score = context_recall("Q?", docs, key_facts=["record-keeping", "retention"])
    assert score == 1.0


def test_context_recall_with_key_facts_empty_list():
    docs = [Document(page_content="Something.")]
    score = context_recall("Q?", docs, key_facts=[])
    assert score == 0.0


def test_context_recall_without_key_facts_uses_sufficiency(sample_docs):
    with patch("eval.metrics.get_context_sufficiency_grader") as m:
        grader = MagicMock()
        grader.invoke.return_value = _mock_grade_suff(True)
        m.return_value = grader
        assert context_recall("Q?", sample_docs, key_facts=None) == 1.0


def test_compute_all_metrics_returns_four_scores(sample_docs):
    with patch("eval.metrics.get_generation_grader") as gm:
        with patch("eval.metrics.get_context_sufficiency_grader") as sm:
            gen_grader = MagicMock()
            gen_grader.invoke.return_value = _mock_grade_gen(True, True)
            gm.return_value = gen_grader
            suff_grader = MagicMock()
            suff_grader.invoke.return_value = _mock_grade_suff(True)
            sm.return_value = suff_grader
            out = compute_all_metrics("Q?", "Answer.", sample_docs, key_facts=["dose"])
            assert set(out.keys()) == {
                "faithfulness",
                "answer_relevance",
                "context_precision",
                "context_recall",
            }
            assert all(0 <= v <= 1 for v in out.values())
