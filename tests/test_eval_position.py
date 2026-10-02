"""Position test (#132): the same chunks, the evidence first or in the middle."""

from langchain_core.documents import Document

from eval.position_test import compare, evidence_indices, placements, vital_score

ITEM = {
    "id": "q",
    "question": "Hvad er dosisgrænsen?",
    "expected_behavior": "answer",
    "nuggets": [
        {"text": "20 mSv", "importance": "vital", "evidence": ["20 mSv pr. år"]},
        {"text": "§ 14", "importance": "okay", "evidence": ["fremgår af bilag 2"]},
    ],
}


def _docs(*texts):
    return [Document(page_content=t, metadata={"source": t[:6]}) for t in texts]


def test_evidence_chunks_are_those_holding_a_vital_quote():
    docs = _docs(
        "IAEA one", "IAEA two", "grænsen er 20 mSv pr. år", "fremgår af bilag 2"
    )

    assert evidence_indices(ITEM, docs) == [2]


def test_middle_is_where_the_first_danish_chunk_sits_today():
    docs = _docs("i1", "i2", "i3", "EVIDENCE", "d2", "d3")

    orders = placements(docs, [3])

    assert [d.page_content for d in orders["first"]] == [
        "EVIDENCE",
        "i1",
        "i2",
        "i3",
        "d2",
        "d3",
    ]
    assert [d.page_content for d in orders["middle"]] == [
        "i1",
        "i2",
        "i3",
        "EVIDENCE",
        "d2",
        "d3",
    ]


def test_several_evidence_chunks_move_together_in_their_own_order():
    docs = _docs("E1", "o1", "o2", "E2", "o3")

    orders = placements(docs, [0, 3])

    assert [d.page_content for d in orders["first"]] == ["E1", "E2", "o1", "o2", "o3"]
    assert [d.page_content for d in orders["middle"]] == ["o1", "o2", "E1", "E2", "o3"]


def test_the_score_is_lenient_vital_recall():
    assert vital_score(ITEM, ["support", "not_support"]) == 1.0
    assert vital_score(ITEM, ["partial_support", "support"]) == 0.5


def test_pairs_are_compared_by_sign_test_over_the_questions_that_differ():
    rows = [
        {"first": 1.0, "middle": 0.5},
        {"first": 1.0, "middle": 0.0},
        {"first": 0.5, "middle": 1.0},
        {"first": 1.0, "middle": 1.0},
    ]

    summary = compare(rows)

    assert summary["first_better"] == 2 and summary["middle_better"] == 1
    assert summary["ties"] == 1
    assert summary["sign_test"]["flipped"] == 3
    assert summary["mean_first"] == 0.875 and summary["mean_middle"] == 0.625


def test_a_run_answers_each_question_twice_and_skips_what_cannot_be_moved(monkeypatch):
    import eval.judge
    import eval.position_test as position_test
    import eval.run_eval
    import graph.chains.generation
    import graph.llm_factory

    retrieved = {
        "Hvad er dosisgrænsen?": _docs("IAEA one", "grænsen er 20 mSv pr. år", "other"),
        "Ingen evidens?": _docs("IAEA one", "other"),
    }
    monkeypatch.setattr(eval.run_eval, "_retrieve_initial", lambda q, ep: retrieved[q])
    monkeypatch.setattr(graph.llm_factory, "get_llm", lambda: object())
    monkeypatch.setattr(
        eval.run_eval, "_judge_llm", lambda g, p, m: (object(), {"judge_model": "j"})
    )

    class Chain:
        def invoke(self, inputs):
            first = inputs["context"].split("\n")[1]
            return "20 mSv" if "20 mSv" in first else "unclear"

    monkeypatch.setattr(
        graph.chains.generation, "get_generation_chain", lambda llm: Chain()
    )
    monkeypatch.setattr(
        eval.judge,
        "judge_item",
        lambda item, answer, context, llm, votes=1: {
            "nuggets": [
                "support" if "20 mSv" in answer else "not_support",
                "not_support",
            ],
            "unsupported_claims": [],
            "refused": False,
        },
    )
    golden = [ITEM, {**ITEM, "id": "none", "question": "Ingen evidens?"}]

    record = position_test.run(golden, "test", delay_sec=0)

    assert record["skipped"]["no_evidence"] == ["none"]
    [row] = record["rows"]
    assert (row["first"], row["middle"], row["evidence_at"]) == (1.0, 0.0, [2])
    assert record["summary"]["first_better"] == 1
