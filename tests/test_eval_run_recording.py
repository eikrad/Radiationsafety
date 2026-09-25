"""run_eval scores each question (scoring v2) and records the run (graph + judge mocked)."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import eval.run_eval as run_eval
from eval.graph_run import load_outputs
from eval.history import load_runs

ANNEX_2 = "For erhvervsmæssig bestråling gælder dosisgrænserne i bilag 2."

GOLDEN = [
    {
        "id": "dk-dose-limits",
        "question": "Hvor findes dosisgrænserne?",
        "topics": ["occupational"],
        "language": "da",
        "source": "dk-law",
        "nuggets": [
            {
                "text": "Dosisgrænserne står i bilag 2",
                "importance": "vital",
                "evidence": ["dosisgrænserne i bilag 2"],
            }
        ],
    },
    {
        "id": "iaea-transport-index",
        "question": "What is the transport index?",
        "topics": ["transport"],
        "language": "en",
        "source": "iaea",
        "nuggets": [
            {
                "text": "Maximum dose rate at 1 m",
                "importance": "vital",
                "evidence": ["maximum dose rate at 1 m"],
            }
        ],
    },
    {
        "id": "out-of-scope-mri",
        "question": "What is the SAR limit for a 3 T MRI?",
        "topics": ["medical"],
        "language": "en",
        "source": "iaea",
        "expected_behavior": "refuse",
    },
]

# Per question: what the judge says. dk passes, transport misses its fact
# (evidence never retrieved -> retrieval_miss), MRI refuses correctly.
VERDICTS = {
    "Hvor findes dosisgrænserne?": {
        "nuggets": ["support"],
        "unsupported_claims": [],
        "refused": False,
    },
    "What is the transport index?": {
        "nuggets": ["not_support"],
        "unsupported_claims": [],
        "refused": False,
    },
    "What is the SAR limit for a 3 T MRI?": {
        "nuggets": [],
        "unsupported_claims": [],
        "refused": True,
    },
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN), encoding="utf-8")

    def fake_graph(question, graph, llm):
        return {
            "generation": f"Answer to {question}",
            "documents": [Document(page_content=ANNEX_2)],
            "context_used_for_generation": ANNEX_2,
            "retrieval_warning": None,
            "web_search_attempted": False,
            "initial_documents": [Document(page_content=ANNEX_2)],
            "sufficient": True,
            "node_path": ["retrieve", "grade_documents", "generate"],
        }

    def fake_judge(item, answer, context, llm):
        return VERDICTS[item["question"]]

    monkeypatch.setattr(run_eval, "_invoke_graph", fake_graph)
    monkeypatch.setattr(run_eval, "judge_item", fake_judge)
    monkeypatch.setattr(
        "graph.llm_factory.get_llm",
        lambda provider=None, model_variant=None, **_: SimpleNamespace(
            model=model_variant or f"{provider or 'gemini'}-model"
        ),
    )
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EVAL_GRADER_PROVIDER", "openai")
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)

    return SimpleNamespace(
        golden=golden,
        history=tmp_path / "history" / "runs.jsonl",
        reports=tmp_path / "reports",
    )


def _run(monkeypatch, ws, *extra: str) -> int:
    argv = [
        "run_eval",
        "--golden",
        str(ws.golden),
        "--output-dir",
        str(ws.reports),
        "--history-file",
        str(ws.history),
        "--no-web-search",
        "--delay-after-graph",
        "0",
        "--delay-between-items",
        "0",
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    return run_eval.main()


def _results(run):
    return {r["id"]: r for r in run["results"]}


def test_a_finished_run_is_recorded_with_its_label_and_settings(monkeypatch, workspace):
    assert _run(monkeypatch, workspace, "--label", "baseline", "--notes", "k=3") == 0

    [run] = load_runs(workspace.history)
    assert run["label"] == "baseline"
    assert run["notes"] == "k=3"
    assert run["dataset"]["n_items"] == 3
    config = run["config"]
    assert config["llm_provider"] == "gemini"
    assert config["llm_model"] == "gemini-model"
    assert config["judge_provider"] == "openai"
    assert config["judge_model"] == "openai-model"
    assert config["judge_is_generator"] is False
    assert config["embedding_model"] == "models/gemini-embedding-001"
    assert config["retriever_k"] == 3
    assert config["web_search"] is False
    assert config["metrics_version"] == 2
    assert "generation.py" in config["prompts"]
    assert run["git"]["commit"]
    assert run["duration_sec"] >= 0


def test_each_question_carries_its_error_type_and_scores(monkeypatch, workspace):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    results = _results(run)
    dk = results["dk-dose-limits"]
    assert (dk["pass"], dk["error_type"]) == (True, "ok")
    assert dk["metrics"]["vital_recall"] == 1.0
    assert dk["metrics"]["evidence_recall_context"] == 1.0
    assert dk["metrics"]["grade_documents_correct"] == 1.0
    # share of answers with an unsupported claim (0/1), so trends stay on a 0-1 scale
    assert dk["metrics"]["unsupported_claim"] == 0.0
    assert dk["unsupported_claims"] == 0
    transport = results["iaea-transport-index"]
    assert (transport["pass"], transport["error_type"]) == (False, "retrieval_miss")
    assert transport["metrics"]["grade_documents_correct"] == 0.0
    mri = results["out-of-scope-mri"]
    assert (mri["pass"], mri["error_type"]) == (True, "ok")
    assert "vital_recall" not in mri["metrics"]
    assert dk["topics"] == ["occupational"]
    assert (transport["language"], transport["source"]) == ("en", "iaea")


def test_the_summary_counts_error_types_and_averages_only_applicable_scores(
    monkeypatch, workspace
):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    summary = run["summary"]
    assert summary["pass_rate"] == pytest.approx(2 / 3)
    assert summary["error_counts"] == {"ok": 2, "retrieval_miss": 1}
    assert summary["judged"] == 3
    # the refusal question has no vital recall, so it does not drag the mean down
    assert summary["vital_recall_mean"] == 0.5


def test_questions_the_judge_could_not_score_are_left_out_of_the_pass_rate(
    monkeypatch, workspace
):
    def judge_fails_on_mri(item, answer, context, llm):
        return None if "MRI" in item["question"] else VERDICTS[item["question"]]

    monkeypatch.setattr(run_eval, "judge_item", judge_fails_on_mri)

    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert run["summary"]["pass_rate"] == 0.5
    assert run["summary"]["judged"] == 2
    assert run["summary"]["error_counts"]["judge_error"] == 1
    assert _results(run)["out-of-scope-mri"]["pass"] is None


def test_full_graph_outputs_are_kept_for_rescoring(monkeypatch, workspace):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    outputs = load_outputs(workspace.reports / f"outputs_{run['run_id']}.json")
    dk = outputs["dk-dose-limits"]
    assert dk["generation"] == "Answer to Hvor findes dosisgrænserne?"
    assert dk["initial_documents"][0].page_content == ANNEX_2
    assert dk["sufficient"] is True


def test_judging_with_the_answering_model_is_recorded_and_warned_about(
    monkeypatch, workspace, capsys
):
    monkeypatch.delenv("EVAL_GRADER_PROVIDER")

    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert run["config"]["judge_is_generator"] is True
    assert "judge is the answering model" in capsys.readouterr().err


def test_the_judge_model_can_be_chosen_separately(monkeypatch, workspace):
    monkeypatch.setenv("EVAL_GRADER_PROVIDER", "gemini")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemini-2.5-pro")

    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert run["config"]["judge_model"] == "gemini-2.5-pro"
    assert run["config"]["judge_is_generator"] is False


def test_a_v1_golden_file_is_refused_with_a_migration_hint(
    monkeypatch, workspace, capsys
):
    workspace.golden.write_text(
        json.dumps([{"id": "old", "question": "Hvad?", "key_facts": ["bilag 2"]}]),
        encoding="utf-8",
    )

    assert _run(monkeypatch, workspace) == 1
    assert "key_facts is the v1 format" in capsys.readouterr().err
    assert not workspace.history.exists()


def test_the_report_carries_the_same_run_header(monkeypatch, workspace):
    _run(monkeypatch, workspace, "--label", "baseline")

    [run] = load_runs(workspace.history)
    report = json.loads((workspace.reports / run["report_file"]).read_text())
    assert report["run_id"] == run["run_id"]
    assert report["label"] == "baseline"
    assert report["config"] == run["config"]
    assert report["dataset"] == run["dataset"]
    markdown = (workspace.reports / run["report_file"]).with_suffix(".md").read_text()
    assert "baseline" in markdown
    assert "retrieval_miss" in markdown


def test_the_report_shows_what_the_judge_flagged(monkeypatch, workspace):
    flagged = dict(VERDICTS)
    flagged["Hvor findes dosisgrænserne?"] = {
        "nuggets": ["support"],
        "unsupported_claims": ["Grænsen er 50 mSv."],
        "refused": False,
    }
    monkeypatch.setattr(
        run_eval, "judge_item", lambda item, answer, context, llm: flagged[item["question"]]
    )
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    report = json.loads((workspace.reports / run["report_file"]).read_text())
    dk = _results(report)["dk-dose-limits"]
    assert dk["judge"]["unsupported_claims"] == ["Grænsen er 50 mSv."]
    assert dk["judge"]["nuggets"] == ["support"]
    markdown = (workspace.reports / run["report_file"]).with_suffix(".md").read_text()
    assert "Grænsen er 50 mSv." in markdown


def test_a_limited_run_is_recorded_with_its_own_question_count(monkeypatch, workspace):
    _run(monkeypatch, workspace, "--limit", "1")

    [run] = load_runs(workspace.history)
    assert run["dataset"]["n_items"] == 1


def test_no_history_leaves_the_history_untouched(monkeypatch, workspace):
    _run(monkeypatch, workspace, "--no-history")

    assert not workspace.history.exists()


def test_a_run_that_crashes_is_not_recorded(monkeypatch, workspace):
    def broken_graph(*args, **kwargs):
        raise RuntimeError("Chroma collection missing")

    monkeypatch.setattr(run_eval, "_invoke_graph", broken_graph)

    with pytest.raises(RuntimeError):
        _run(monkeypatch, workspace)
    assert not workspace.history.exists()


def test_a_recorded_run_refreshes_the_dashboard(monkeypatch, workspace):
    _run(monkeypatch, workspace, "--label", "baseline")

    page = (workspace.reports / "dashboard.html").read_text(encoding="utf-8")
    [run] = load_runs(workspace.history)
    assert run["run_id"] in page


# --- re-scoring saved runs ----------------------------------------------------


def test_a_saved_run_can_be_rescored_without_running_the_graph(monkeypatch, workspace):
    _run(monkeypatch, workspace, "--label", "baseline")
    [original] = load_runs(workspace.history)

    def graph_must_not_run(*args, **kwargs):
        raise AssertionError("rescoring must not run the graph")

    def stricter_judge(item, answer, context, llm):
        verdict = dict(VERDICTS[item["question"]])
        if item["id"] == "dk-dose-limits":
            verdict["unsupported_claims"] = ["Grænsen er 50 mSv"]
        return verdict

    monkeypatch.setattr(run_eval, "_invoke_graph", graph_must_not_run)
    monkeypatch.setattr(run_eval, "judge_item", stricter_judge)
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gpt-4o")

    assert _run(monkeypatch, workspace, "--rescore", original["run_id"]) == 0

    original, rescored = load_runs(workspace.history)
    assert rescored["rescored_from"] == original["run_id"]
    assert rescored["label"] == f"rescore of {original['run_id']}"
    # the answers are the original run's: its models, retrieval and prompts
    for key in ("llm_model", "retriever_k", "prompts", "embedding_model"):
        assert rescored["config"][key] == original["config"][key]
    assert rescored["git"] == original["git"]
    # the judgement is new
    assert rescored["config"]["judge_model"] == "gpt-4o"
    assert _results(rescored)["dk-dose-limits"]["error_type"] == "unsupported_claim"


def test_rescoring_an_unknown_run_says_so(monkeypatch, workspace, capsys):
    assert _run(monkeypatch, workspace, "--rescore", "20990101_000000") == 1
    assert "no saved outputs for run 20990101_000000" in capsys.readouterr().err


def test_questions_added_after_the_run_are_skipped_when_rescoring(
    monkeypatch, workspace, capsys
):
    _run(monkeypatch, workspace, "--limit", "2")
    [original] = load_runs(workspace.history)

    _run(monkeypatch, workspace, "--rescore", original["run_id"])

    _, rescored = load_runs(workspace.history)
    assert [r["id"] for r in rescored["results"]] == [
        "dk-dose-limits",
        "iaea-transport-index",
    ]
    assert rescored["dataset"]["n_items"] == 2
    assert "out-of-scope-mri: not in the saved run" in capsys.readouterr().err
