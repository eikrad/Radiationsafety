"""run_eval scores each question (scoring v2) and records the run (graph + judge mocked)."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import eval.run_eval as run_eval
from eval.graph_run import load_outputs
from eval.history import load_runs

ANNEX_2 = "For erhvervsmæssig bestråling gælder dosisgrænserne i bilag 2."
FILLER = "Unrelated chunk about packaging."

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
    import ingestion

    # The fake retrievals below are built for the graph retrieving 3 per collection.
    monkeypatch.setattr(ingestion, "RETRIEVER_K", 3)
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

    def fake_judge(item, answer, context, llm, votes=1):
        return VERDICTS[item["question"]]

    monkeypatch.setattr(run_eval, "_invoke_graph", fake_graph)
    monkeypatch.setattr(run_eval, "judge_item", fake_judge)
    monkeypatch.setattr(
        run_eval,
        "git_info",
        lambda: {"commit": "abc1234", "branch": "staging", "dirty": False},
    )
    monkeypatch.setattr(
        "ingestion.index_fingerprint",
        lambda ep: {"radiation-dk-law": {"chunks": 2, "content_hash": "c0ffee"}},
    )
    monkeypatch.setattr(
        "graph.llm_factory.get_llm",
        lambda provider=None, model_variant=None, **_: SimpleNamespace(
            model=model_variant or f"{provider or 'gemini'}-model"
        ),
    )
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    # pinned: these tests are about recording, not about the default embeddings
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EVAL_GRADER_PROVIDER", "openai")
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_VOTES", raising=False)

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
    def judge_fails_on_mri(item, answer, context, llm, votes=1):
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


def test_the_judge_votes_as_often_as_configured_and_the_run_records_it(
    monkeypatch, workspace
):
    asked_with = []

    def counting_judge(item, answer, context, llm, votes=1):
        asked_with.append(votes)
        return VERDICTS[item["question"]]

    monkeypatch.setattr(run_eval, "judge_item", counting_judge)
    monkeypatch.setenv("EVAL_JUDGE_VOTES", "5")
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert run["config"]["judge_votes"] == 5
    assert set(asked_with) == {5}


def test_the_judge_votes_three_times_by_default(monkeypatch, workspace):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert run["config"]["judge_votes"] == 3


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
        run_eval,
        "judge_item",
        lambda item, answer, context, llm, votes=1: flagged[item["question"]],
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

    def stricter_judge(item, answer, context, llm, votes=1):
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


# --- Retrieval-only runs: compare embeddings without answering or judging -----


@pytest.fixture
def retrieval_only(monkeypatch, workspace):
    """Every question retrieves the Danish annex, ranked second in its collection."""
    asked = []

    def fake_retrieve(question, embedding_provider):
        asked.append((question, embedding_provider))
        # the graph's top 3 per collection, merged and deduplicated
        return [Document(page_content=FILLER), Document(page_content=ANNEX_2)]

    def must_not_run(*_, **__):
        raise AssertionError("retrieval-only runs neither answer nor judge")

    def fake_ranked(question, embedding_provider, depth):
        return {
            "iaea": [Document(page_content=FILLER)] * depth,
            "dk": [Document(page_content=FILLER), Document(page_content=ANNEX_2)]
            + [Document(page_content=FILLER)] * (depth - 2),
        }

    monkeypatch.setattr(run_eval, "_retrieve_initial", fake_retrieve)
    monkeypatch.setattr(run_eval, "_retrieve_ranked", fake_ranked)
    monkeypatch.setattr(run_eval, "_invoke_graph", must_not_run)
    monkeypatch.setattr(run_eval, "judge_item", must_not_run)
    return SimpleNamespace(asked=asked)


def test_a_retrieval_only_run_scores_the_first_retrieval_without_answering_or_judging(
    monkeypatch, workspace, retrieval_only
):
    assert _run(monkeypatch, workspace, "--retrieval-only", "--label", "emb") == 0

    [run] = load_runs(workspace.history)
    results = _results(run)
    assert results["dk-dose-limits"]["metrics"]["evidence_recall_initial"] == 1.0
    assert results["iaea-transport-index"]["metrics"]["evidence_recall_initial"] == 0.0
    # the refusal question has no evidence to find
    assert results["out-of-scope-mri"]["metrics"] == {}
    assert run["summary"]["evidence_recall_initial_mean"] == 0.5
    assert run["summary"]["pass_rate"] is None
    assert [q for q, _ in retrieval_only.asked] == [g["question"] for g in GOLDEN[:2]]


def test_a_retrieval_only_run_records_the_embedding_settings(
    monkeypatch, workspace, retrieval_only
):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    monkeypatch.setenv("EMBED_QUERY_INSTRUCTION", "false")
    _run(monkeypatch, workspace, "--retrieval-only")

    [run] = load_runs(workspace.history)
    config = run["config"]
    assert config["retrieval_only"] is True
    assert config["embedding_provider"] == "scaleway"
    assert config["embedding_model"] == "bge-multilingual-gemma2"
    assert config["embedding_query_instruction"] is False
    assert config["retriever_k"] == 3
    assert "llm_model" not in config and "judge_model" not in config
    assert {ep for _, ep in retrieval_only.asked} == {"scaleway"}


def test_a_full_run_records_whether_questions_carried_an_instruction(
    monkeypatch, workspace
):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    # Gemini embeddings take no instruction
    assert run["config"]["embedding_query_instruction"] is False
    assert run["config"]["retrieval_only"] is False


def test_a_retrieval_only_run_refuses_when_privacy_mode_overrides_the_embeddings(
    monkeypatch, workspace, retrieval_only, capsys
):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")

    assert _run(monkeypatch, workspace, "--retrieval-only") == 1
    assert "LLM_PROVIDER=ollama" in capsys.readouterr().err
    assert retrieval_only.asked == []
    assert load_runs(workspace.history) == []


def test_a_retrieval_only_run_ranks_the_evidence_in_a_deeper_retrieval(
    monkeypatch, workspace, retrieval_only
):
    _run(monkeypatch, workspace, "--retrieval-only", "--depth", "5")

    [run] = load_runs(workspace.history)
    metrics = _results(run)["dk-dose-limits"]["metrics"]
    assert metrics["evidence_recall_at_1"] == 0.0
    assert metrics["evidence_recall_at_3"] == 1.0
    assert metrics["evidence_recall_at_5"] == 1.0
    assert "evidence_recall_at_10" not in metrics
    assert metrics["reciprocal_rank"] == 0.5
    assert run["summary"]["reciprocal_rank_mean"] == 0.25  # transport: not found
    assert _results(run)["dk-dose-limits"]["evidence_ranks"] == [2]


def test_recall_at_a_text_budget_uses_the_budget_given(
    monkeypatch, workspace, retrieval_only
):
    one_filler = str(len(FILLER))  # the annex, ranked second, does not fit
    _run(monkeypatch, workspace, "--retrieval-only", "--char-budget", one_filler)

    [run] = load_runs(workspace.history)
    assert _results(run)["dk-dose-limits"]["metrics"]["evidence_recall_budget"] == 0.0
    assert run["config"]["char_budget"] == len(FILLER)
    assert run["config"]["retrieval_depth"] == 20


def test_a_retrieval_only_run_keeps_the_ranked_lists_for_rescoring(
    monkeypatch, workspace, retrieval_only
):
    _run(monkeypatch, workspace, "--retrieval-only", "--depth", "3")

    [run] = load_runs(workspace.history)
    [outputs_file] = workspace.reports.glob("outputs_*.json")
    saved = load_outputs(outputs_file)["dk-dose-limits"]
    assert [d.page_content for d in saved["ranked_dk"]] == [FILLER, ANNEX_2, FILLER]
    assert len(saved["ranked_iaea"]) == 3


def test_a_deep_retrieval_that_disagrees_with_the_graph_is_reported(
    monkeypatch, workspace, retrieval_only, capsys
):
    """Recall@3 from the deep list must mean what the graph retrieves at k=3."""
    monkeypatch.setattr(
        run_eval,
        "_retrieve_initial",
        lambda question, embedding_provider: [Document(page_content="other chunk")],
    )
    _run(monkeypatch, workspace, "--retrieval-only")

    [run] = load_runs(workspace.history)
    assert run["summary"]["top_k_mismatches"] == [
        "dk-dose-limits",
        "iaea-transport-index",
    ]
    assert "top 3 differs from the graph" in capsys.readouterr().err


def test_a_retrieval_only_run_can_be_rescored_after_evidence_was_added(
    monkeypatch, workspace, retrieval_only
):
    """Pooling: a newly confirmed quote counts without retrieving again."""
    _run(monkeypatch, workspace, "--retrieval-only")
    [first] = load_runs(workspace.history)

    golden = json.loads(workspace.golden.read_text(encoding="utf-8"))
    golden[1]["nuggets"][0]["evidence"].append("unrelated chunk about packaging")
    workspace.golden.write_text(json.dumps(golden), encoding="utf-8")

    def must_not_retrieve(*_, **__):
        raise AssertionError("rescoring reuses the saved retrieval")

    monkeypatch.setattr(run_eval, "_retrieve_initial", must_not_retrieve)
    monkeypatch.setattr(run_eval, "_retrieve_ranked", must_not_retrieve)
    assert _run(monkeypatch, workspace, "--rescore", first["run_id"]) == 0

    rescored = load_runs(workspace.history)[-1]
    assert rescored["config"]["retrieval_only"] is True
    assert rescored["rescored_from"] == first["run_id"]
    transport = _results(rescored)["iaea-transport-index"]["metrics"]
    assert transport["evidence_recall_at_1"] == 1.0


# --- Guards: what a recorded run can be trusted to describe -------------------


def test_a_run_records_the_search_index_it_retrieved_from(monkeypatch, workspace):
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert run["config"]["index"] == {
        "radiation-dk-law": {"chunks": 2, "content_hash": "c0ffee"}
    }


def test_a_retrieval_only_run_records_the_search_index_too(
    monkeypatch, workspace, retrieval_only
):
    _run(monkeypatch, workspace, "--retrieval-only")

    [run] = load_runs(workspace.history)
    assert run["config"]["index"]["radiation-dk-law"]["chunks"] == 2


@pytest.fixture
def dirty(monkeypatch):
    monkeypatch.setattr(
        run_eval,
        "git_info",
        lambda: {"commit": "abc1234", "branch": "staging", "dirty": True},
    )


def test_a_run_on_uncommitted_code_is_not_recorded_by_default(
    monkeypatch, workspace, dirty, capsys
):
    assert _run(monkeypatch, workspace) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    assert load_runs(workspace.history) == []


def test_a_run_on_uncommitted_code_can_be_recorded_on_purpose(
    monkeypatch, workspace, dirty
):
    assert _run(monkeypatch, workspace, "--allow-dirty") == 0

    [run] = load_runs(workspace.history)
    assert run["git"]["dirty"] is True


def test_a_debugging_run_without_history_may_use_uncommitted_code(
    monkeypatch, workspace, dirty
):
    assert _run(monkeypatch, workspace, "--no-history") == 0


def test_a_retrieval_shallower_than_the_graphs_is_not_flagged_as_disagreeing(
    monkeypatch, workspace, retrieval_only
):
    _run(monkeypatch, workspace, "--retrieval-only", "--depth", "2")

    [run] = load_runs(workspace.history)
    assert run["summary"]["top_k_mismatches"] == []


def test_a_full_run_reports_progress_for_every_question(monkeypatch, workspace, capsys):
    """A run of 39 questions with pauses takes a while; it must not look stuck."""
    _run(monkeypatch, workspace)

    err = capsys.readouterr().err
    assert "3 questions" in err
    assert "[1/3] answered dk-dose-limits" in err
    assert "[3/3] answered out-of-scope-mri" in err
    assert "[2/3] judged iaea-transport-index: retrieval_miss" in err


def test_the_run_header_names_the_embeddings(monkeypatch, workspace, capsys):
    _run(monkeypatch, workspace)

    assert "embeddings = gemini/models/gemini-embedding-001" in capsys.readouterr().err


def test_a_warning_shown_with_an_answer_is_counted(monkeypatch, workspace):
    answered = run_eval._invoke_graph

    def warned(question, graph, llm):
        return {**answered(question, graph, llm), "retrieval_warning": "not verified"}

    monkeypatch.setattr(run_eval, "_invoke_graph", warned)
    _run(monkeypatch, workspace)

    [run] = load_runs(workspace.history)
    assert _results(run)["dk-dose-limits"]["metrics"]["warning_shown"] == 1.0
    assert run["summary"]["warning_shown_mean"] == 1.0


# --- Regrading: the sufficiency grader alone, on a saved run's retrievals ------


@pytest.fixture
def regrade(monkeypatch, workspace):
    """A saved full run, then a grader that calls anything mentioning bilag 2 sufficient."""
    _run(monkeypatch, workspace)
    [run] = load_runs(workspace.history)
    seen = []

    def fake_grader(question, documents, llm):
        seen.append((question, [d.page_content for d in documents]))
        return any("bilag 2" in d.page_content for d in documents)

    def must_not_run(*_, **__):
        raise AssertionError("regrading neither answers nor judges")

    monkeypatch.setattr(run_eval, "_grade_sufficient", fake_grader)
    monkeypatch.setattr(run_eval, "_invoke_graph", must_not_run)
    monkeypatch.setattr(run_eval, "judge_item", must_not_run)
    return SimpleNamespace(run_id=run["run_id"], seen=seen)


def test_regrading_scores_the_grader_on_the_saved_first_retrievals(
    monkeypatch, workspace, regrade
):
    assert _run(monkeypatch, workspace, "--regrade", regrade.run_id) == 0

    run = load_runs(workspace.history)[-1]
    results = _results(run)
    assert run["config"]["grader_only"] is True
    assert run["regraded_from"] == regrade.run_id
    assert results["dk-dose-limits"]["metrics"]["grade_documents_correct"] == 1.0
    # the annex was retrieved for the transport question too: "sufficient" is wrong
    assert results["iaea-transport-index"]["metrics"]["grade_documents_correct"] == 0.0
    assert results["out-of-scope-mri"]["metrics"]["grade_documents_correct"] == 0.0


def test_regrading_removes_the_evidence_to_test_a_retrieval_that_misses_it(
    monkeypatch, workspace, regrade
):
    """Labelled insufficient cases (#129): the same retrieval without its evidence chunks."""
    _run(monkeypatch, workspace, "--regrade", regrade.run_id)

    run = load_runs(workspace.history)[-1]
    dk = _results(run)["dk-dose-limits"]["metrics"]
    assert dk["grade_documents_ablation_correct"] == 1.0
    assert ("Hvor findes dosisgrænserne?", []) in regrade.seen
    # only retrievals that held all their evidence can be ablated
    assert (
        "grade_documents_ablation_correct"
        not in _results(run)["iaea-transport-index"]["metrics"]
    )
    assert run["summary"]["grade_documents_ablation_correct_mean"] == 1.0
