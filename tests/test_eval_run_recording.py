"""run_eval records each finished run in the history (graph, LLM and metrics mocked)."""

import json
from types import SimpleNamespace

import pytest

import eval.run_eval as run_eval
from eval.history import load_runs

GOLDEN = [
    {
        "id": "dk-dose-limits",
        "question": "Hvor findes dosisgrænserne?",
        "key_facts": ["bilag 2"],
        "topics": ["occupational"],
        "language": "da",
        "source": "dk-law",
    },
    {
        "id": "iaea-transport-index",
        "question": "What is the transport index?",
        "topics": ["transport"],
        "language": "en",
        "source": "iaea",
    },
]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN), encoding="utf-8")

    def fake_graph(question, graph, llm):
        return {
            "generation": f"Answer to {question}",
            "documents": [],
            "context_used_for_generation": "",
            "retrieval_warning": None,
            "web_search_attempted": False,
        }

    def fake_metrics(**kwargs):
        recall = 1.0 if "dosis" in kwargs["question"] else 0.0
        return {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "context_precision": 1.0,
            "context_recall": recall,
        }

    monkeypatch.setattr(run_eval, "_invoke_graph", fake_graph)
    monkeypatch.setattr("eval.metrics.compute_all_metrics", fake_metrics)
    monkeypatch.setattr(
        "graph.llm_factory.get_llm",
        lambda provider=None, **_: SimpleNamespace(model="fake-model"),
    )
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("EVAL_GRADER_PROVIDER", raising=False)

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


def test_a_finished_run_is_recorded_with_its_label_and_settings(monkeypatch, workspace):
    assert _run(monkeypatch, workspace, "--label", "baseline", "--notes", "k=3") == 0

    [run] = load_runs(workspace.history)
    assert run["label"] == "baseline"
    assert run["notes"] == "k=3"
    assert run["dataset"]["n_items"] == 2
    config = run["config"]
    assert config["llm_provider"] == "gemini"
    assert config["llm_model"] == "fake-model"
    assert config["grader_model"] == "fake-model"
    assert config["embedding_model"] == "models/gemini-embedding-001"
    assert config["retriever_k"] == 3
    assert config["web_search"] is False
    assert config["pass_rule"] == "all"
    assert config["metrics_version"] >= 1
    assert "generation.py" in config["prompts"]
    assert run["git"]["commit"]
    assert run["duration_sec"] >= 0


def test_recorded_results_carry_scores_and_topic_tags(monkeypatch, workspace):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    dk, iaea = run["results"]
    assert (dk["pass"], iaea["pass"]) == (True, False)
    assert dk["topics"] == ["occupational"]
    assert (iaea["language"], iaea["source"]) == ("en", "iaea")
    assert run["summary"]["pass_rate"] == 0.5


def test_the_report_carries_the_same_run_header(monkeypatch, workspace):
    _run(monkeypatch, workspace, "--label", "baseline")

    [run] = load_runs(workspace.history)
    report = json.loads((workspace.reports / run["report_file"]).read_text())
    assert report["run_id"] == run["run_id"]
    assert report["label"] == "baseline"
    assert report["config"] == run["config"]
    assert report["dataset"] == run["dataset"]
    assert report["git"] == run["git"]
    markdown = (workspace.reports / run["report_file"]).with_suffix(".md")
    assert "baseline" in markdown.read_text()


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
