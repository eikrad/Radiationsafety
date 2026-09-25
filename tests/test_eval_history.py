"""Tests for eval run history: fingerprints, run records, the history file."""

from eval.history import append_run, build_run_record, dataset_fingerprint, load_runs

GOLDEN = [
    {
        "id": "dose-limits",
        "question": "Hvor findes dosisgrænserne?",
        "expected_answer": "I bilag 2.",
        "key_facts": ["bilag 2"],
        "topics": ["occupational"],
    },
    {
        "id": "transport-index",
        "question": "What is the transport index?",
        "key_facts": ["maximum dose rate at 1 m"],
        "topics": ["transport"],
    },
]


def test_runs_over_the_same_questions_are_comparable():
    same_questions = [dict(item) for item in GOLDEN]
    assert dataset_fingerprint(GOLDEN) == dataset_fingerprint(same_questions)


def test_changed_grading_targets_are_flagged_but_runs_stay_comparable():
    edited = [dict(item) for item in GOLDEN]
    edited[0]["key_facts"] = ["bilag 2", "20 mSv"]

    before, after = dataset_fingerprint(GOLDEN), dataset_fingerprint(edited)

    assert after["questions_hash"] == before["questions_hash"]
    assert after["content_hash"] != before["content_hash"]


def test_retagging_topics_does_not_break_comparability():
    retagged = [dict(item) for item in GOLDEN]
    retagged[0]["topics"] = ["medical"]

    assert dataset_fingerprint(retagged) == dataset_fingerprint(GOLDEN)


def test_a_limited_run_is_not_comparable_with_the_full_set():
    assert (
        dataset_fingerprint(GOLDEN[:1])["questions_hash"]
        != dataset_fingerprint(GOLDEN)["questions_hash"]
    )


def _result(item_id: str, passed: bool, **extra) -> dict:
    return {
        "id": item_id,
        "question": f"question {item_id}",
        "pass": passed,
        "metrics": {"faithfulness": 1.0, "context_recall": 0.5 if passed else 0.0},
        "generation_preview": "A long generated answer …",
        "retrieval_warning": None,
        "web_search_attempted": False,
        **extra,
    }


def _record(run_id: str, **overrides) -> dict:
    fields = {
        "run_id": run_id,
        "summary": {"pass_rate": 0.5},
        "results": [
            _result("a", True, topics=["medical"], language="da", source="dk-law"),
            _result("b", False),
        ],
        "dataset": dataset_fingerprint(GOLDEN),
        "config": {"llm_model": "gemini-2.5-flash", "retriever_k": 3},
        "git": {"commit": "abc1234", "branch": "staging", "dirty": False},
        "label": "baseline",
    }
    fields.update(overrides)
    return build_run_record(**fields)


def test_a_recorded_run_reads_back_with_scores_and_tags(tmp_path):
    history = tmp_path / "runs.jsonl"

    append_run(_record("20260925_101500"), history)
    [run] = load_runs(history)

    assert run["label"] == "baseline"
    assert run["timestamp"] == "2026-09-25T10:15:00+00:00"
    assert run["config"]["retriever_k"] == 3
    first, second = run["results"]
    assert first["pass"] is True
    assert first["metrics"]["context_recall"] == 0.5
    assert (first["topics"], first["language"], first["source"]) == (
        ["medical"],
        "da",
        "dk-law",
    )
    assert second["topics"] == []


def test_history_keeps_scores_but_not_generated_answers(tmp_path):
    history = tmp_path / "runs.jsonl"

    append_run(_record("20260925_101500"), history)

    assert "A long generated answer" not in history.read_text(encoding="utf-8")


def test_history_reads_back_oldest_first(tmp_path):
    history = tmp_path / "runs.jsonl"
    append_run(_record("20260925_120000"), history)
    append_run(_record("20260916_094008"), history)

    assert [r["run_id"] for r in load_runs(history)] == [
        "20260916_094008",
        "20260925_120000",
    ]


def test_a_corrupt_history_line_does_not_hide_the_other_runs(tmp_path):
    history = tmp_path / "runs.jsonl"
    append_run(_record("20260916_094008"), history)
    with open(history, "a", encoding="utf-8") as f:
        f.write('{"run_id": "half-writ\n')
    append_run(_record("20260925_120000"), history)

    assert len(load_runs(history)) == 2


def test_no_history_file_means_no_runs(tmp_path):
    assert load_runs(tmp_path / "missing.jsonl") == []
