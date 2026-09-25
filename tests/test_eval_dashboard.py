"""Tests for the eval dashboard's data: grouping, comparisons, topic breakdown."""

import json

from eval.dashboard import dashboard_data
from eval.history import build_run_record

METRICS = ("faithfulness", "answer_relevance", "context_precision", "context_recall")

QUESTIONS = {
    "dk-dose-limits": {
        "topics": ["occupational"],
        "language": "da",
        "source": "dk-law",
    },
    "dk-xray-license": {"topics": ["medical"], "language": "da", "source": "dk-law"},
    "en-transport-index": {"topics": ["transport"], "language": "en", "source": "iaea"},
}

CONFIG = {
    "llm_model": "gemini-2.5-flash",
    "retriever_k": 3,
    "metrics_version": 1,
    "prompts": {"generation.py": "aaa", "missing_query_chain.py": "bbb"},
}


def _run(
    run_id: str,
    passes: dict[str, bool],
    *,
    label=None,
    config=None,
    questions_hash="set-a",
    content_hash="content-1",
    report_file=None,
):
    results = []
    for qid, passed in passes.items():
        score = 1.0 if passed else 0.0
        results.append(
            {
                "id": qid,
                "question": f"Question {qid}?",
                **QUESTIONS.get(qid, {}),
                "pass": passed,
                "metrics": dict.fromkeys(METRICS, score),
            }
        )
    n = len(results)
    summary = {"pass_rate": sum(passes.values()) / n if n else 0.0}
    for m in METRICS:
        summary[f"{m}_mean"] = sum(r["metrics"][m] for r in results) / n
    return build_run_record(
        run_id=run_id,
        summary=summary,
        results=results,
        dataset={
            "questions_hash": questions_hash,
            "content_hash": content_hash,
            "n_items": n,
        },
        config=CONFIG if config is None else config,
        git={"commit": "abc1234", "branch": "staging", "dirty": False},
        label=label,
        report_file=report_file,
    )


ALL_PASS = dict.fromkeys(QUESTIONS, True)


def _set(data, questions_hash="set-a"):
    return next(s for s in data["sets"] if s["questions_hash"] == questions_hash)


def test_no_runs_gives_an_empty_dashboard():
    data = dashboard_data([])

    assert data["sets"] == []
    assert data["runs"] == {}


def test_runs_on_different_question_sets_never_share_a_trend():
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260920_100000", {"dk-dose-limits": True}, questions_hash="set-b"),
        _run("20260925_101500", ALL_PASS),
    ]

    data = dashboard_data(runs)

    assert _set(data, "set-a")["run_ids"] == ["20260916_094008", "20260925_101500"]
    assert _set(data, "set-b")["run_ids"] == ["20260920_100000"]
    assert [s["questions_hash"] for s in data["sets"]] == ["set-a", "set-b"]


def test_a_question_that_stopped_passing_is_a_regression_since_the_previous_run():
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260925_101500", {**ALL_PASS, "dk-xray-license": False}),
    ]

    comparison = _set(dashboard_data(runs))["comparisons"]["20260925_101500"][
        "previous"
    ]

    assert comparison["base_run_id"] == "20260916_094008"
    assert [q["id"] for q in comparison["regressions"]] == ["dk-xray-license"]
    assert comparison["improvements"] == []
    assert comparison["summary_delta"]["pass_rate"] < 0
    assert comparison["verdict"] == "1 regression, 0 improvements"


def test_the_first_run_of_a_set_has_nothing_to_compare_against():
    data = dashboard_data([_run("20260916_094008", ALL_PASS)])

    assert _set(data)["comparisons"]["20260916_094008"] == {
        "previous": None,
        "baseline": None,
    }


def test_runs_are_compared_against_the_latest_run_labelled_baseline():
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260917_090000", ALL_PASS, label="baseline"),
        _run("20260920_090000", ALL_PASS),
        _run("20260925_101500", {**ALL_PASS, "en-transport-index": False}),
    ]

    s = _set(dashboard_data(runs))

    assert s["baseline_run_id"] == "20260917_090000"
    against = s["comparisons"]["20260925_101500"]["baseline"]
    assert against["base_run_id"] == "20260917_090000"
    assert [q["id"] for q in against["regressions"]] == ["en-transport-index"]
    assert s["comparisons"]["20260917_090000"]["baseline"] is None


def test_a_baseline_can_be_chosen_explicitly():
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260917_090000", ALL_PASS, label="baseline"),
        _run("20260925_101500", ALL_PASS),
    ]

    s = _set(dashboard_data(runs, baseline_run_id="20260916_094008"))

    assert s["baseline_run_id"] == "20260916_094008"


def test_without_a_labelled_baseline_the_first_run_is_the_baseline():
    runs = [_run("20260916_094008", ALL_PASS), _run("20260925_101500", ALL_PASS)]

    assert _set(dashboard_data(runs))["baseline_run_id"] == "20260916_094008"


def test_a_comparison_names_only_the_settings_that_changed():
    changed = {
        **CONFIG,
        "retriever_k": 5,
        "prompts": {**CONFIG["prompts"], "generation.py": "ccc"},
    }
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260925_101500", ALL_PASS, config=changed),
    ]

    comparison = _set(dashboard_data(runs))["comparisons"]["20260925_101500"][
        "previous"
    ]

    assert comparison["config_diff"] == [
        {"field": "prompts.generation.py", "before": "aaa", "after": "ccc"},
        {"field": "retriever_k", "before": 3, "after": 5},
    ]


def test_comparing_with_an_imported_run_says_its_settings_are_unknown():
    runs = [
        _run("20260916_094008", ALL_PASS, config={}),
        _run("20260925_101500", ALL_PASS),
    ]

    comparison = _set(dashboard_data(runs))["comparisons"]["20260925_101500"][
        "previous"
    ]

    assert comparison["config_diff"] is None
    assert any("not recorded" in w for w in comparison["warnings"])


def test_changed_grading_targets_are_marked_and_warned_about():
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260925_101500", ALL_PASS, content_hash="content-2"),
    ]

    s = _set(dashboard_data(runs))

    assert s["markers"] == [
        {"run_id": "20260925_101500", "reasons": ["grading targets changed"]}
    ]
    comparison = s["comparisons"]["20260925_101500"]["previous"]
    assert "grading targets changed" in comparison["warnings"]


def test_a_scoring_change_is_marked_and_warned_about():
    rescored = {**CONFIG, "metrics_version": 2}
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260925_101500", ALL_PASS, config=rescored),
    ]

    s = _set(dashboard_data(runs))

    assert s["markers"][0]["reasons"] == ["scoring changed"]
    assert (
        "scoring changed" in s["comparisons"]["20260925_101500"]["previous"]["warnings"]
    )


def test_topic_breakdown_counts_questions_and_flags_thin_topics():
    passes = {**ALL_PASS, "en-transport-index": False}
    data = dashboard_data([_run("20260925_101500", passes)])

    breakdown = _set(data)["breakdown"]["20260925_101500"]

    assert breakdown["topics"]["medical"] == {
        "n": 1,
        "pass_rate": 1.0,
        "means": dict.fromkeys(METRICS, 1.0),
        "low_n": True,
    }
    assert breakdown["topics"]["transport"]["pass_rate"] == 0.0
    assert breakdown["language"]["da"]["n"] == 2
    assert breakdown["source"]["iaea"]["n"] == 1


def test_untagged_questions_are_grouped_as_untagged():
    run = _run("20260916_094008", {"q-old": True, "q-old-2": False})

    breakdown = _set(dashboard_data([run]))["breakdown"]["20260916_094008"]

    assert breakdown["topics"]["untagged"]["n"] == 2
    assert breakdown["language"]["untagged"]["n"] == 2


def test_answer_previews_come_from_local_reports_when_present(tmp_path):
    report = {
        "run_id": "20260925_101500",
        "results": [{"id": "dk-dose-limits", "generation_preview": "I bilag 2."}],
    }
    (tmp_path / "report_20260925_101500.json").write_text(json.dumps(report))
    runs = [
        _run(
            "20260925_101500",
            ALL_PASS,
            report_file="report_20260925_101500.json",
        ),
        _run("20260926_101500", ALL_PASS, report_file="report_20260926_101500.json"),
    ]

    data = dashboard_data(runs, reports_dir=tmp_path)

    present = data["runs"]["20260925_101500"]["results"]["dk-dose-limits"]
    assert present["preview"] == "I bilag 2."
    missing = data["runs"]["20260926_101500"]["results"]["dk-dose-limits"]
    assert missing["preview"] is None


def test_the_grid_lists_each_questions_outcome_per_run():
    runs = [
        _run("20260916_094008", ALL_PASS),
        _run("20260925_101500", {**ALL_PASS, "dk-xray-license": False}),
    ]

    data = dashboard_data(runs)
    s = _set(data)

    assert [q["id"] for q in s["questions"]] == list(QUESTIONS)
    assert s["questions"][1]["topics"] == ["medical"]
    assert (
        data["runs"]["20260925_101500"]["results"]["dk-xray-license"]["pass"] is False
    )
