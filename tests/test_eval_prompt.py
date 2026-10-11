"""Prompt test (#145): the saved contexts of a full run, answered with the previous
and the revised generation prompt."""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

EVIDENCE = "grænsen er 20 mSv pr. år"
ANSWER = {
    "id": "q",
    "question": "Hvad er dosisgrænsen?",
    "language": "da",
    "expected_behavior": "answer",
    "nuggets": [{"text": "20 mSv", "importance": "vital", "evidence": [EVIDENCE]}],
}
REFUSE = {
    "id": "r",
    "question": "What is the SAR limit for MRI?",
    "language": "en",
    "expected_behavior": "refuse",
    "nuggets": [],
}


def _saved_run(context):
    return {
        "context_used_for_generation": context,
        "initial_documents": [Document(page_content=context)],
        "sufficient": True,
        "retrieval_warning": None,
    }


OUTPUTS = {
    "q": _saved_run(f"[Source: BEK 1384 (Danish law)]\n{EVIDENCE}"),
    "r": _saved_run("[Source: https://mri.example (web)]\nSAR 2 W/kg"),
}


def _fake_models(monkeypatch, answers):
    """A generator that answers by system prompt, and a judge that records its calls.

    answers: {system prompt name: {question id: answer}}.
    """
    import eval.judge
    import eval.prompt_test as prompt_test
    import eval.run_eval
    import graph.chains.generation
    import graph.llm_factory

    seen = {"contexts": [], "judged": []}
    by_question = {i["question"]: i["id"] for i in (ANSWER, REFUSE)}

    def generate(prompt_value):
        system, human = prompt_value.to_messages()
        name = "previous" if system.content == prompt_test.PREVIOUS else "revised"
        qid = next(i for q, i in by_question.items() if q in human.content)
        seen["contexts"].append(human.content)
        return answers[name][qid]

    def judge(item, answer, context, llm, votes=1):
        seen["judged"].append((item["id"], answer, context, votes))
        return {
            "nuggets": ["support" if "20 mSv" in answer else "not_support"]
            * len(item["nuggets"]),
            "unsupported_claims": ["made up"] if "W/kg" in answer else [],
            "refused": "not covered" in answer,
        }

    # the machinery is under test, not today's prompt, which may equal PREVIOUS
    monkeypatch.setattr(graph.chains.generation, "RAG_SYSTEM", "Revised prompt.")
    monkeypatch.setattr(graph.llm_factory, "get_llm", lambda: RunnableLambda(generate))
    monkeypatch.setattr(
        eval.run_eval, "_judge_llm", lambda g, p, m: (object(), {"judge_model": "j"})
    )
    monkeypatch.setattr(eval.judge, "judge_item", judge)
    return seen


def test_each_saved_context_is_answered_with_both_prompts_and_judged_with_3_votes(
    monkeypatch,
):
    import eval.prompt_test as prompt_test

    seen = _fake_models(
        monkeypatch,
        {
            "previous": {"q": "20 mSv", "r": "The SAR limit is 2 W/kg."},
            "revised": {"q": "20 mSv", "r": "MRI is not covered by these documents."},
        },
    )

    record = prompt_test.run([ANSWER, REFUSE], OUTPUTS, "test", delay_sec=0)

    answer_row, refuse_row = record["rows"]
    assert answer_row["previous"]["pass"] and answer_row["revised"]["pass"]
    assert refuse_row["previous"]["error_type"] == "missed_refusal"
    assert refuse_row["revised"]["error_type"] == "ok"
    assert all(
        OUTPUTS["q"]["context_used_for_generation"] in c for c in seen["contexts"][:2]
    )
    assert {(qid, ctx, votes) for qid, _, ctx, votes in seen["judged"]} == {
        ("q", OUTPUTS["q"]["context_used_for_generation"], 3),
        ("r", OUTPUTS["r"]["context_used_for_generation"], 3),
    }
    assert record["config"]["judge_votes"] == 3
    assert record["summary"]["passed"] == {"previous": 1, "revised": 2}
    assert record["answers"]["r"]["revised"] == "MRI is not covered by these documents."


def test_questions_missing_from_the_saved_run_or_unjudged_are_listed_not_scored(
    monkeypatch,
):
    import eval.judge
    import eval.prompt_test as prompt_test

    _fake_models(
        monkeypatch,
        {"previous": {"q": "20 mSv"}, "revised": {"q": "20 mSv"}},
    )
    monkeypatch.setattr(eval.judge, "judge_item", lambda *a, **k: None)

    record = prompt_test.run([ANSWER, REFUSE], {"q": OUTPUTS["q"]}, "test", 0)

    assert record["rows"] == []
    assert record["skipped"] == {"not_in_run": ["r"], "judge_error": ["q"]}


def _row(qid, behavior, previous, revised):
    """previous/revised: (error_type, lenient vital recall, unsupported claims)."""

    def scores(error_type, recall, claims):
        return {
            "pass": error_type == "ok",
            "error_type": error_type,
            "vital_recall_lenient": recall,
            "refused": error_type in ("ok", "wrong_refusal") and recall is None,
            "unsupported_claims": claims,
        }

    return {
        "id": qid,
        "expected_behavior": behavior,
        "previous": scores(*previous),
        "revised": scores(*revised),
    }


def test_the_summary_checks_the_adoption_rule_set_before_measuring():
    from eval.prompt_test import summarize

    rows = [
        _row("same", "answer", ("ok", 1.0, []), ("ok", 1.0, [])),
        _row("grounded", "answer", ("unsupported_claim", 1.0, ["x"]), ("ok", 1.0, [])),
        _row("fact lost", "answer", ("ok", 1.0, []), ("generator_miss", 0.5, [])),
        _row("mri", "refuse", ("missed_refusal", None, []), ("ok", None, [])),
    ]

    summary = summarize(rows)

    assert summary["passed"] == {"previous": 2, "revised": 3}
    assert summary["unsupported_answers"] == {"previous": 1, "revised": 0}
    assert summary["clean_refusals"] == {"previous": 0, "revised": 1, "of": 1}
    assert summary["wrong_refusals"] == {"previous": 0, "revised": 0}
    assert summary["vital_recall"]["previous_better"] == 1
    assert summary["rule"] == {
        "fewer_unsupported_answers": True,
        "vital_recall_held": True,
        "every_refusal_clean": True,
        "no_more_wrong_refusals": True,
    }
    assert summary["rule_met"] is True

    another_loss = _row(
        "lost too", "answer", ("ok", 1.0, []), ("generator_miss", 0.5, [])
    )
    assert summarize(rows + [another_loss])["rule"]["vital_recall_held"] is False
    assert summarize(rows + [another_loss])["rule_met"] is False


def _main(monkeypatch, tmp_path, dirty=False, extra=()):
    import json

    import eval.history
    import eval.prompt_test as prompt_test
    from eval.graph_run import save_outputs

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([ANSWER, REFUSE]), encoding="utf-8")
    outputs = tmp_path / "outputs_20261003_132636.json"
    save_outputs(OUTPUTS, outputs)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("EVAL_DELAY_AFTER_GRAPH_SEC", "0")
    monkeypatch.setattr(
        eval.history,
        "git_info",
        lambda: {"commit": "abc1234", "branch": "feature", "dirty": dirty},
    )
    _fake_models(
        monkeypatch,
        {
            "previous": {"q": "20 mSv", "r": "The SAR limit is 2 W/kg."},
            "revised": {"q": "20 mSv", "r": "MRI is not covered by these documents."},
        },
    )
    log, reports = tmp_path / "prompt_tests.jsonl", tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt_test",
            "--golden",
            str(golden),
            "--outputs",
            str(outputs),
            "--label",
            "prompt-145",
            "--log",
            str(log),
            "--reports",
            str(reports),
            *extra,
        ],
    )
    return prompt_test.main(), log, reports


def test_a_prompt_test_records_where_its_contexts_came_from_and_reports_the_answers(
    monkeypatch, tmp_path
):
    import json

    code, log, reports = _main(monkeypatch, tmp_path)

    assert code == 0
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["label"] == "prompt-145"
    assert record["config"]["contexts_from"] == "outputs_20261003_132636.json"
    assert "answers" not in record
    (report,) = reports.glob("prompt_test_*.json")
    answers = json.loads(report.read_text())["answers"]
    assert answers["r"]["revised"] == "MRI is not covered by these documents."


def test_a_prompt_test_refuses_to_run_on_uncommitted_code(monkeypatch, tmp_path):
    code, log, _ = _main(monkeypatch, tmp_path, dirty=True)

    assert code == 1
    assert not log.exists()


def test_saved_answers_are_judged_again_without_generating(monkeypatch):
    import eval.prompt_test as prompt_test
    import graph.llm_factory

    seen = _fake_models(monkeypatch, {"previous": {}, "revised": {}})

    def must_not_generate(prompt_value):
        raise AssertionError("a re-judged run must not generate")

    monkeypatch.setattr(
        graph.llm_factory, "get_llm", lambda: RunnableLambda(must_not_generate)
    )
    saved_answers = {
        "q": {"previous": "20 mSv", "revised": "20 mSv"},
        "r": {"previous": "The SAR limit is 2 W/kg.", "revised": "MRI is not covered."},
    }

    record = prompt_test.run(
        [ANSWER, REFUSE], OUTPUTS, "rejudged", 0, answers=saved_answers
    )

    assert record["summary"]["passed"] == {"previous": 1, "revised": 2}
    assert {answer for _, answer, _, _ in seen["judged"]} == {
        "20 mSv",
        "The SAR limit is 2 W/kg.",
        "MRI is not covered.",
    }
    assert record["answers"] == saved_answers


def _report(tmp_path, contexts_from="outputs_20261003_132636.json"):
    import json

    report = tmp_path / "prompt_test_20261010_215307.json"
    report.write_text(
        json.dumps(
            {
                "run_id": "20261010_215307",
                "config": {"llm_model": "gemma-4", "contexts_from": contexts_from},
                "answers": {
                    "q": {"previous": "20 mSv", "revised": "20 mSv"},
                    "r": {"previous": "2 W/kg.", "revised": "MRI is not covered."},
                },
            }
        ),
        encoding="utf-8",
    )
    return report


def test_a_rejudged_prompt_test_records_the_run_and_model_it_came_from(
    monkeypatch, tmp_path
):
    import json

    code, log, _ = _main(
        monkeypatch, tmp_path, extra=("--rejudge", str(_report(tmp_path)))
    )

    assert code == 0
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["config"]["rejudged_from"] == "20261010_215307"
    assert record["config"]["llm_model"] == "gemma-4"
    assert record["summary"]["passed"] == {"previous": 1, "revised": 2}


def test_rejudging_against_other_contexts_than_the_answers_had_is_refused(
    monkeypatch, tmp_path
):
    report = _report(tmp_path, contexts_from="outputs_20260101_000000.json")

    code, log, _ = _main(monkeypatch, tmp_path, extra=("--rejudge", str(report)))

    assert code == 1
    assert not log.exists()


def test_the_record_describes_the_code_as_it_was_when_the_test_started(monkeypatch):
    import eval.history
    import eval.prompt_test as prompt_test
    import graph.llm_factory

    _fake_models(monkeypatch, {"previous": {}, "revised": {}})
    edited = []

    def generate_while_someone_edits(prompt_value):
        edited.append(True)
        return "20 mSv"

    monkeypatch.setattr(
        graph.llm_factory,
        "get_llm",
        lambda: RunnableLambda(generate_while_someone_edits),
    )
    monkeypatch.setattr(
        eval.history,
        "git_info",
        lambda: {"commit": "abc1234", "branch": "b", "dirty": bool(edited)},
    )

    record = prompt_test.run([ANSWER], OUTPUTS, "test", 0)

    assert record["git"]["dirty"] is False
