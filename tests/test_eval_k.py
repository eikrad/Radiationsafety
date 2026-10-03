"""k test (step 5): the same question answered from the top 3 or the top 5 per collection."""

from langchain_core.documents import Document

from eval.k_test import contexts, stratum
from eval.position_test import answer_score, compare

ANSWER = {
    "id": "q",
    "question": "Hvad er dosisgrænsen?",
    "expected_behavior": "answer",
    "nuggets": [
        {"text": "20 mSv", "importance": "vital", "evidence": ["20 mSv pr. år"]},
        {"text": "6 mSv", "importance": "vital", "evidence": ["6 mSv for lærlinge"]},
    ],
}
REFUSE = {"id": "r", "question": "MRI?", "expected_behavior": "refuse", "nuggets": []}


def _docs(prefix, n):
    return [Document(page_content=f"{prefix}{i}", id=f"{prefix}{i}") for i in range(n)]


def test_each_k_takes_the_top_k_of_each_collection_iaea_first():
    iaea, dk = _docs("i", 5), _docs("d", 5)

    by_k = contexts(iaea, dk, (3, 5))

    assert [d.id for d in by_k[3]] == ["i0", "i1", "i2", "d0", "d1", "d2"]
    assert len(by_k[5]) == 10


def test_questions_are_grouped_by_where_their_evidence_is():
    small = [Document(page_content="grænsen er 20 mSv pr. år")]
    large = small + [Document(page_content="6 mSv for lærlinge")]

    assert stratum(ANSWER, small, large) == "evidence only at k=5"
    assert stratum(ANSWER, large, large) == "evidence at k=3"
    assert stratum(ANSWER, small, small) == "evidence missing at k=5"
    assert stratum(REFUSE, small, large) == "should refuse"


def test_a_question_to_refuse_scores_1_only_for_a_clean_refusal():
    clean = {"nuggets": [], "unsupported_claims": [], "refused": True}

    assert answer_score(REFUSE, clean) == 1.0
    assert answer_score(REFUSE, {**clean, "refused": False}) == 0.0
    assert answer_score(REFUSE, {**clean, "unsupported_claims": ["x"]}) == 0.0
    assert (
        answer_score(ANSWER, {**clean, "nuggets": ["support", "partial_support"]})
        == 0.75
    )


def test_two_variants_are_compared_under_their_own_names():
    rows = [{"k3": 1.0, "k5": 0.5}, {"k3": 0.5, "k5": 1.0}, {"k3": 1.0, "k5": 1.0}]

    summary = compare(rows, "k3", "k5")

    assert summary["k3_better"] == 1 and summary["k5_better"] == 1
    assert summary["ties"] == 1


def test_the_retriever_k_can_be_set_without_a_code_change(monkeypatch):
    import pytest

    from ingestion import _retriever_k

    monkeypatch.delenv("RETRIEVER_K", raising=False)
    assert _retriever_k() == 5
    monkeypatch.setenv("RETRIEVER_K", "3")
    assert _retriever_k() == 3
    monkeypatch.setenv("RETRIEVER_K", "0")
    with pytest.raises(ValueError, match="RETRIEVER_K"):
        _retriever_k()


def test_a_run_answers_each_question_from_both_contexts(monkeypatch):
    import eval.judge
    import eval.k_test as k_test
    import eval.run_eval
    import graph.chains.generation
    import graph.llm_factory

    iaea = [Document(page_content=f"IAEA {i}") for i in range(5)]
    dk = [Document(page_content=f"dk {i}") for i in range(3)] + [
        Document(page_content="grænsen er 20 mSv pr. år"),
        Document(page_content="6 mSv for lærlinge"),
    ]
    monkeypatch.setattr(
        eval.run_eval, "_retrieve_ranked", lambda q, ep, depth: {"iaea": iaea, "dk": dk}
    )
    monkeypatch.setattr(graph.llm_factory, "get_llm", lambda: object())
    monkeypatch.setattr(
        eval.run_eval, "_judge_llm", lambda g, p, m: (object(), {"judge_model": "j"})
    )

    class Chain:
        def invoke(self, inputs):
            return "20 mSv og 6 mSv" if "20 mSv" in inputs["context"] else "ved ikke"

    monkeypatch.setattr(
        graph.chains.generation, "get_generation_chain", lambda llm: Chain()
    )
    monkeypatch.setattr(
        eval.judge,
        "judge_item",
        lambda item, answer, context, llm, votes=1: {
            "nuggets": ["support" if "mSv" in answer else "not_support"]
            * len(item["nuggets"]),
            "unsupported_claims": [],
            "refused": "ved ikke" in answer,
        },
    )

    record = k_test.run([ANSWER, REFUSE], "test", delay_sec=0)

    answer_row, refuse_row = record["rows"]
    assert (answer_row["stratum"], answer_row["k3"], answer_row["k5"]) == (
        "evidence only at k=5",
        0.0,
        1.0,
    )
    assert refuse_row["stratum"] == "should refuse"
    assert (refuse_row["k3"], refuse_row["k5"]) == (1.0, 0.0)
    assert record["summary"]["all"]["k5_better"] == 1
