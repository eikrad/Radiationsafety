"""Saved runs can be reopened without a model, and failures cannot outrun history."""

import gzip
import json

import pytest
from langchain_core.documents import Document

from eval.history import append_run, load_runs
from eval.saved_runs import SavedRuns, load_outputs, save_outputs

RUN_ID = "20261118_120000"
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
        "config": {"llm_model": "original"},
        "git": {"commit": "original-commit"},
        "summary": {"pass_rate": 1.0},
        "results": [{"id": "question", "pass": True}],
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
    assert elsewhere.judged_passes(RUN_ID) == {"question": True}


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
    assert runs.judged_passes(RUN_ID) is None


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
    append_run(report(config={"llm_model": "first"}), runs.history_file)
    append_run(
        report(
            config={"llm_model": "latest"},
            results=[
                {"id": "question", "pass": False},
                {"id": "unjudged", "pass": None},
            ],
        ),
        runs.history_file,
    )
    assert runs.provenance(RUN_ID)["config"] == {"llm_model": "first"}
    assert runs.judged_passes(RUN_ID) == {"question": False}


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
