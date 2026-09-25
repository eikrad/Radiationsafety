"""Tests for eval run history: fingerprints, run records, the history file."""

import json
import subprocess

import pytest

from eval.history import (
    append_run,
    build_run_record,
    dataset_fingerprint,
    git_info,
    import_reports,
    load_runs,
    prompt_fingerprints,
)

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


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "staging")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "generation.py").write_text("system = 'v1'\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_run_on_committed_code_names_its_commit_and_branch(repo):
    info = git_info(repo)

    assert info["branch"] == "staging"
    assert len(info["commit"]) >= 7
    assert info["dirty"] is False


def test_a_run_on_uncommitted_changes_is_marked_dirty(repo):
    (repo / "generation.py").write_text("system = 'v2'\n")

    assert git_info(repo)["dirty"] is True


def test_untracked_files_do_not_mark_a_run_dirty(repo):
    (repo / "notes.txt").write_text("scratch")

    assert git_info(repo)["dirty"] is False


def test_outside_a_git_repo_the_run_is_still_recorded_without_git_info(tmp_path):
    assert git_info(tmp_path) == {"commit": None, "branch": None, "dirty": None}


def test_editing_one_prompt_changes_only_its_fingerprint(tmp_path):
    (tmp_path / "generation.py").write_text("system = 'v1'\n")
    (tmp_path / "missing_query_chain.py").write_text("system = 'q1'\n")
    before = prompt_fingerprints(tmp_path)

    (tmp_path / "generation.py").write_text("system = 'v2'\n")
    after = prompt_fingerprints(tmp_path)

    assert set(after) == {"generation.py", "missing_query_chain.py"}
    assert after["generation.py"] != before["generation.py"]
    assert after["missing_query_chain.py"] == before["missing_query_chain.py"]


def _old_report(reports, run_id: str, n_questions: int = 2) -> None:
    """A report as run_eval wrote it before runs carried a header."""
    results = [_result(f"q{i}", True) for i in range(n_questions)]
    payload = {"summary": {"pass_rate": 1.0}, "results": results, "run_id": run_id}
    reports.mkdir(exist_ok=True)
    (reports / f"report_{run_id}.json").write_text(json.dumps(payload))


def test_old_reports_since_a_date_are_backfilled(tmp_path):
    reports, history = tmp_path / "reports", tmp_path / "runs.jsonl"
    _old_report(reports, "20260310_101657", n_questions=1)
    _old_report(reports, "20260916_094008", n_questions=13)
    _old_report(reports, "20260916_104838", n_questions=13)

    assert import_reports(reports, history, since="20260916") == 2

    runs = load_runs(history)
    assert [r["run_id"] for r in runs] == ["20260916_094008", "20260916_104838"]
    first = runs[0]
    assert first["label"] == "imported"
    assert first["report_file"] == "report_20260916_094008.json"
    assert first["dataset"]["n_items"] == 13
    assert first["dataset"]["content_hash"] is None
    assert first["git"] is None
    assert runs[0]["dataset"]["questions_hash"] == runs[1]["dataset"]["questions_hash"]


def test_backfilling_twice_does_not_duplicate_runs(tmp_path):
    reports, history = tmp_path / "reports", tmp_path / "runs.jsonl"
    _old_report(reports, "20260916_094008")

    import_reports(reports, history)

    assert import_reports(reports, history) == 0
    assert len(load_runs(history)) == 1


def test_a_report_with_a_run_header_keeps_it_when_backfilled(tmp_path):
    reports, history = tmp_path / "reports", tmp_path / "runs.jsonl"
    record = _record("20260925_101500")
    reports.mkdir()
    (reports / "report_20260925_101500.json").write_text(json.dumps(record))

    import_reports(reports, history)

    [run] = load_runs(history)
    assert run["label"] == "baseline"
    assert run["config"]["retriever_k"] == 3
    assert run["dataset"] == record["dataset"]
