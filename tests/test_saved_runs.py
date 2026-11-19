"""Saved runs can be reopened without a model, and failures cannot outrun history."""

import gzip
import json

import pytest
from langchain_core.documents import Document

from eval.history import append_run, load_runs
from eval.saved_runs import (
    SavedRunProvenanceError,
    SavedRuns,
    load_outputs,
    save_outputs,
)

RUN_ID = "20261118_120000"
QUESTIONS = [{"id": "question", "question": "Original question?"}]
OUTPUTS = {
    "question": {
        "generation": "Dosisgrænser",
        "documents": [Document(page_content="context", metadata={"page": 3})],
        "initial_documents": [Document(page_content="first")],
        "ranked_iaea": [Document(page_content="IAEA", metadata={"rank": 1})],
        "ranked_dk": [Document(page_content="dansk")],
        "sufficient": False,
        "node_path": ["retrieve", "generate"],
    }
}


@pytest.fixture
def runs(tmp_path):
    return SavedRuns(tmp_path / "reports", tmp_path / "history" / "runs.jsonl")


def report(**changes):
    return {
        "run_id": RUN_ID,
        "dataset": {"questions_hash": "questions"},
        "config": {"llm_provider": "gemini", "llm_model": "original"},
        "git": {"commit": "original-commit"},
        "summary": {"pass_rate": 1.0},
        "results": [{"id": "question", "question": "Original question?", "pass": True}],
        **changes,
    }


def test_finished_answers_reopen_with_original_provenance_in_another_workspace(
    runs, tmp_path
):
    runs.save(RUN_ID, OUTPUTS)
    artifacts = runs.finish(report(), "# Report")

    assert artifacts.json_report.name == f"report_{RUN_ID}.json"
    assert json.loads(artifacts.json_report.read_text()) == report()
    assert artifacts.markdown_report.read_text() == "# Report"
    assert (
        artifacts.archive == runs.history_file.parent / "outputs" / f"{RUN_ID}.json.gz"
    )
    raw = (runs.output_dir / f"outputs_{RUN_ID}.json").read_bytes()
    assert artifacts.archive.read_bytes() == gzip.compress(
        raw, compresslevel=9, mtime=0
    )
    elsewhere = SavedRuns(tmp_path / "other-reports", runs.history_file)
    assert elsewhere.load(RUN_ID) == OUTPUTS
    assert elsewhere.provenance(RUN_ID)["config"] == report()["config"]
    assert elsewhere.provenance(RUN_ID)["git"] == report()["git"]
    assert elsewhere.regrade_labels(RUN_ID, QUESTIONS) == {"question": True}


@pytest.mark.parametrize("blocked", ["json", "markdown", "archive"])
def test_failed_completion_keeps_previous_history_and_recoverable_answers(
    runs, blocked
):
    append_run(report(run_id="20261117_120000"), runs.history_file)
    before = runs.history_file.read_bytes()
    runs.save(RUN_ID, OUTPUTS)
    paths = {
        "json": runs.output_dir / f"report_{RUN_ID}.json",
        "markdown": runs.output_dir / f"report_{RUN_ID}.md",
        "archive": runs.history_file.parent / "outputs" / f"{RUN_ID}.json.gz",
    }
    paths[blocked].mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        runs.finish(report(), "# Report")

    assert runs.history_file.read_bytes() == before
    assert runs.load(RUN_ID) == OUTPUTS
    if blocked == "archive":
        assert json.loads(paths["json"].read_text()) == report()
        assert paths["markdown"].read_text() == "# Report"
    else:
        assert not paths["archive"].exists()


def test_missing_answers_prevent_recording_a_full_run(runs):
    with pytest.raises(FileNotFoundError):
        runs.finish(report(), "# Report")
    assert load_runs(runs.history_file) == []


def test_failed_history_append_retains_reports_and_archive(runs):
    runs.save(RUN_ID, OUTPUTS)
    runs.history_file.mkdir(parents=True)
    with pytest.raises(IsADirectoryError):
        runs.finish(report(), "# Report")
    archive = runs.history_file.parent / "outputs" / f"{RUN_ID}.json.gz"
    assert load_outputs(archive) == OUTPUTS
    assert (runs.output_dir / f"report_{RUN_ID}.json").is_file()
    assert (runs.output_dir / f"report_{RUN_ID}.md").is_file()


@pytest.mark.parametrize(
    "changes",
    [
        {"rescored_from": "original"},
        {"regraded_from": "original"},
        {"config": {"retrieval_only": True}},
    ],
)
def test_derived_and_retrieval_only_runs_record_without_new_answers(runs, changes):
    artifacts = runs.finish(report(**changes), "# Report")
    assert artifacts.archive is None
    assert len(load_runs(runs.history_file)) == 1
    assert not (runs.history_file.parent / "outputs").exists()


def test_no_history_keeps_only_local_artifacts(runs):
    runs.save(RUN_ID, OUTPUTS)
    artifacts = runs.finish(report(), "# Report", record_history=False)
    assert artifacts.archive is None
    assert not runs.history_file.parent.exists()
    assert runs.load(RUN_ID) == OUTPUTS
    assert runs.provenance(RUN_ID) == report()
    with pytest.raises(SavedRunProvenanceError, match="not recorded in history"):
        runs.regrade_labels(RUN_ID, QUESTIONS)


def test_local_answers_take_priority_and_corruption_is_not_hidden(runs):
    runs.save(RUN_ID, OUTPUTS)
    artifacts = runs.finish(report(), "# Report")
    runs.save(RUN_ID, {"changed": {"generation": "local"}})
    assert runs.load(RUN_ID)["changed"]["generation"] == "local"
    (runs.output_dir / f"outputs_{RUN_ID}.json").write_text("invalid json")
    with pytest.raises(json.JSONDecodeError):
        runs.load(RUN_ID)
    assert load_outputs(artifacts.archive) == OUTPUTS


def test_provenance_uses_first_history_entry_but_labels_use_latest(runs):
    runs.finish(report(), "# Report", record_history=False)
    first_config = {"llm_provider": "gemini", "llm_model": "first"}
    append_run(report(config=first_config), runs.history_file)
    append_run(
        report(
            config={"llm_model": "latest"},
            results=[
                {"id": "question", "question": "Original question?", "pass": False},
                {"id": "unjudged", "pass": None},
            ],
        ),
        runs.history_file,
    )
    assert runs.provenance(RUN_ID)["config"] == first_config
    assert runs.regrade_labels(RUN_ID, QUESTIONS) == {"question": False}


def test_missing_provenance_is_unknown_and_derived_ids_are_not_followed(runs):
    runs.save(RUN_ID, OUTPUTS)
    assert runs.provenance(RUN_ID) == {}
    runs.finish(report(), "# Report")
    runs.finish(report(run_id="20261118_120001", rescored_from=RUN_ID), "# Rescore")
    assert runs.load("20261118_120001") is None
    assert runs.load("missing") is None


def test_standalone_outputs_supply_absent_document_fields(tmp_path):
    path = tmp_path / "outputs.json"
    save_outputs({"failed": {"error": "provider failure"}}, path)
    assert load_outputs(path) == {
        "failed": {
            "error": "provider failure",
            "documents": [],
            "initial_documents": [],
            "ranked_iaea": [],
            "ranked_dk": [],
        }
    }


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"run_id": "another-run"}, "different run"),
        ({"config": None}, "original configuration"),
        ({"config": {}}, "original configuration"),
        ({"config": {"llm_model": "original"}}, "original llm_provider"),
        (
            {"config": {"llm_provider": "gemini", "llm_model": " "}},
            "original llm_model",
        ),
        ({"config": {"retrieval_only": "false"}}, "invalid original retrieval_only"),
        ({"git": None}, "original git commit"),
        ({"git": {"commit": ""}}, "original git commit"),
        ({"results": None}, "missing original questions"),
        ({"results": []}, "missing or ambiguous original question"),
        (
            {"results": [{"id": "question", "question": ""}]},
            "missing or ambiguous original question",
        ),
        (
            {"results": [{"id": "question", "question": "Different?"}]},
            "question text changed",
        ),
        (
            {"results": [{"id": "question", "question": "Original question?"}] * 2},
            "ambiguous original question",
        ),
    ],
)
def test_rescore_rejects_incomplete_or_mismatched_metadata(runs, changes, message):
    artifacts = runs.finish(report(), "# Report", record_history=False)
    artifacts.json_report.write_text(json.dumps(report(**changes)))
    with pytest.raises(SavedRunProvenanceError, match=message):
        runs.rescore_provenance(
            RUN_ID, [{"id": "question", "question": "Original question?"}]
        )


def test_rescore_reports_corrupt_metadata_and_empty_selection(runs):
    artifacts = runs.finish(report(), "# Report", record_history=False)
    with pytest.raises(SavedRunProvenanceError, match="no saved questions match"):
        runs.rescore_provenance(RUN_ID, [])
    artifacts.json_report.write_text("invalid json")
    with pytest.raises(SavedRunProvenanceError, match="not valid JSON"):
        runs.rescore_provenance(
            RUN_ID, [{"id": "question", "question": "Original question?"}]
        )


def test_rescore_preserves_history_priority_instead_of_hiding_incomplete_metadata(runs):
    runs.finish(report(), "# Report", record_history=False)
    append_run(report(config={}), runs.history_file)
    with pytest.raises(SavedRunProvenanceError, match="original configuration"):
        runs.rescore_provenance(
            RUN_ID, [{"id": "question", "question": "Original question?"}]
        )


def test_rescore_accepts_selected_original_questions_with_changed_grading_criteria(
    runs,
):
    original = report(
        results=[
            {"id": "one", "question": "First?"},
            {"id": "two", "question": "Second?"},
            {"id": "removed", "question": "Not selected?"},
        ]
    )
    runs.finish(original, "# Report", record_history=False)
    questions = [
        {"id": "two", "question": "Second?", "nuggets": [{"text": "new fact"}]},
        {"id": "one", "question": "First?", "expected_behavior": "refuse"},
    ]
    assert runs.rescore_provenance(RUN_ID, questions) == original


def test_rescore_validates_retrieval_runs_without_requiring_an_answer_model(runs):
    original = report(config={"retrieval_only": True, "embedding_model": "original"})
    runs.finish(original, "# Report", record_history=False)
    questions = [{"id": "question", "question": "Original question?"}]
    assert runs.rescore_provenance(RUN_ID, questions) == original


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rescored_from": "unrelated"}, "not a rescore of"),
        ({"rescored_from": None}, "not a rescore of"),
        ({"regraded_from": RUN_ID}, "regrade does not supply judge labels"),
        ({"config": {"grader_only": True}}, "regrade does not supply judge labels"),
        ({"config": None}, "missing label run configuration"),
        ({"results": None}, "missing original questions"),
        (
            {"results": [{"id": "question", "pass": True}]},
            "ambiguous original question",
        ),
        (
            {"results": [{"id": "question", "question": "Changed?", "pass": True}]},
            "question text changed",
        ),
        ({"results": report()["results"] * 2}, "ambiguous original question"),
        (
            {"results": [{**report()["results"][0], "pass": "false"}]},
            "label must be true, false, or null",
        ),
        (
            {"results": [{**report()["results"][0], "pass": 1}]},
            "label must be true, false, or null",
        ),
    ],
)
def test_regrade_rejects_unrelated_or_invalid_label_records(runs, changes, message):
    append_run(report(), runs.history_file)
    label = report(run_id="labels", rescored_from=RUN_ID)
    append_run({**label, **changes}, runs.history_file)
    with pytest.raises(SavedRunProvenanceError, match=message):
        runs.regrade_labels(RUN_ID, QUESTIONS, labels_from="labels")


def test_regrade_validates_latest_label_record_without_falling_back(runs):
    append_run(report(), runs.history_file)
    label = report(run_id="labels", rescored_from=RUN_ID)
    append_run(label, runs.history_file)
    append_run({**label, "rescored_from": "other-answer-run"}, runs.history_file)
    with pytest.raises(SavedRunProvenanceError, match="not a rescore of"):
        runs.regrade_labels(RUN_ID, QUESTIONS, labels_from="labels")


def test_regrade_partial_labels_keep_only_selected_boolean_verdicts(runs):
    questions = QUESTIONS + [
        {"id": "unjudged", "question": "Not judged?"},
        {"id": "omitted", "question": "Not rescored?"},
    ]
    append_run(report(results=questions), runs.history_file)
    append_run(
        report(
            run_id="labels",
            rescored_from=RUN_ID,
            results=[
                {**questions[0], "pass": False},
                {**questions[1], "pass": None},
                {"id": "not-selected", "question": "Another?", "pass": True},
            ],
        ),
        runs.history_file,
    )
    assert runs.regrade_labels(RUN_ID, questions, labels_from="labels") == {
        "question": False
    }


def test_regrade_requires_source_metadata_even_with_valid_label_lineage(runs):
    append_run(report(run_id="labels", rescored_from=RUN_ID), runs.history_file)
    with pytest.raises(
        SavedRunProvenanceError, match="Cannot regrade.*missing original metadata"
    ):
        runs.regrade_labels(RUN_ID, QUESTIONS, labels_from="labels")


def test_regrade_allows_sufficiency_checks_on_unjudged_retrieval_outputs(runs):
    append_run(
        report(
            config={"retrieval_only": True},
            results=[
                {**QUESTIONS[0], "pass": None},
            ],
        ),
        runs.history_file,
    )
    assert runs.regrade_labels(RUN_ID, QUESTIONS) == {}
