"""Does the revised generation prompt answer better than the previous one? (#145)

In the full run `k5` (2026-10-03, 32/39) 6 of the 7 failures happened after
retrieval: answers that changed the wording of a legal condition (unsupported
claims), a fact left out, and two questions on non-ionising radiation answered
instead of refused. This test replays the context each question's generator
saw in a saved full run, web results included, and answers it twice: with the
generation prompt from before #145 and with today's. Only the prompt differs.
Both answers are scored as in a full run (eval.scoring.score_item), with 3
groundedness votes: unsupported claims are the outcome, and a single vote flips
on about 3 questions per run.

Trust-Score found that prompting alone does not stop models from answering
from their own knowledge, so the effect is measured, not assumed.

    uv run python -m eval.prompt_test \\
        --outputs eval/reports/outputs_20261003_132636.json --label prompt-145
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from eval.position_test import compare
from eval.scoring import score_item

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = _PROJECT_ROOT / "eval" / "history" / "prompt_tests.jsonl"
DEFAULT_REPORTS = _PROJECT_ROOT / "eval" / "reports"
JUDGE_VOTES = 3

# graph/chains/generation.py RAG_SYSTEM before #145, verbatim (typo included)
PREVIOUS = """You are an assistant for IAEA and Danish radiation safety documents. Use the following retrieved context to answer the question. Each context block is labeled with [Source: ...]; sources may be document names or web URLs. Use and cite web sources (URLs) when they are relevant to the question. If you cannot find the answer in the context, say so. Be concise and cite the source when relevant. Uso only information from the retrieved context"""
NAMES = ("previous", "revised")


def summarize(rows: list[dict]) -> dict:
    """Both prompts side by side, and the adoption rule set before measuring
    (ROADMAP, #145): fewer answers with an unsupported claim; on lenient vital
    recall the revised prompt scores lower on at most 1 more answerable question
    than it scores higher; every question to refuse refused without an
    unsupported claim; no more refusals of questions whose evidence was in the
    context (wrong_refusal)."""
    answerable = [r for r in rows if r["expected_behavior"] == "answer"]
    to_refuse = [r for r in rows if r["expected_behavior"] != "answer"]

    def count(selected, test):
        return {name: sum(bool(test(r[name])) for r in selected) for name in NAMES}

    unsupported = count(rows, lambda s: s["unsupported_claims"])
    clean = count(to_refuse, lambda s: s["pass"])
    wrong = count(answerable, lambda s: s["error_type"] == "wrong_refusal")
    recall = compare(
        [{n: r[n]["vital_recall_lenient"] for n in NAMES} for r in answerable],
        *NAMES,
    )
    rule = {
        "fewer_unsupported_answers": unsupported["revised"] < unsupported["previous"],
        "vital_recall_held": recall["previous_better"] <= recall["revised_better"] + 1,
        "every_refusal_clean": clean["revised"] == len(to_refuse),
        "no_more_wrong_refusals": wrong["revised"] <= wrong["previous"],
    }
    return {
        "questions": len(rows),
        "passed": count(rows, lambda s: s["pass"]),
        "unsupported_answers": unsupported,
        "clean_refusals": {**clean, "of": len(to_refuse)},
        "wrong_refusals": wrong,
        "vital_recall": recall,
        "error_types": {
            n: dict(Counter(r[n]["error_type"] for r in rows)) for n in NAMES
        },
        "rule": rule,
        "rule_met": all(rule.values()),
    }


def _answer_and_score(item, saved, system, generator, judge, delay_sec) -> dict | None:
    from eval.judge import judge_item
    from eval.run_eval import _invoke_with_retry
    from graph.chains.generation import get_generation_chain

    context = saved["context_used_for_generation"]
    answer = _invoke_with_retry(
        get_generation_chain(generator, system).invoke,
        {"context": context, "chat_history": "", "question": item["question"]},
    )
    time.sleep(delay_sec)
    verdict = judge_item(item, answer, context, judge, votes=JUDGE_VOTES)
    time.sleep(delay_sec)
    if verdict is None:
        return None
    scores = score_item(
        item,
        {
            "initial_documents": saved.get("initial_documents") or [],
            "context": context,
            "sufficient": saved.get("sufficient"),
            "retrieval_warning": saved.get("retrieval_warning"),
        },
        verdict,
    )
    return {
        "answer": answer,
        "scores": {
            "pass": scores["pass"],
            "error_type": scores["error_type"],
            "vital_recall_lenient": scores["vital_recall_lenient"],
            "refused": scores["refused"],
            "unsupported_claims": list(verdict["unsupported_claims"]),
        },
    }


def run(golden: list[dict], outputs: dict[str, dict], label, delay_sec: float) -> dict:
    """Answer every golden question in outputs (a saved full run, see
    eval.graph_run.load_outputs) with both prompts and score both answers."""
    from eval.history import git_info
    from eval.run_eval import _judge_llm, _model_name
    from graph.chains.generation import RAG_SYSTEM
    from graph.llm_factory import DEFAULT_PROVIDER, get_llm

    generator = get_llm()
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    judge, judge_info = _judge_llm(generator, provider, _model_name(generator))
    prompts = {"previous": PREVIOUS, "revised": RAG_SYSTEM}

    rows, answers = [], {}
    skipped = {"not_in_run": [], "judge_error": []}
    for item in golden:
        saved = outputs.get(item["id"])
        if saved is None:
            skipped["not_in_run"].append(item["id"])
            continue
        results = {
            name: _answer_and_score(item, saved, system, generator, judge, delay_sec)
            for name, system in prompts.items()
        }
        scored = {name: r for name, r in results.items() if r is not None}
        if len(scored) < len(results):
            skipped["judge_error"].append(item["id"])
            continue
        rows.append(
            {
                "id": item["id"],
                "language": item.get("language"),
                "expected_behavior": item["expected_behavior"],
                **{name: r["scores"] for name, r in scored.items()},
            }
        )
        answers[item["id"]] = {name: r["answer"] for name, r in scored.items()}
        print(
            f"  [{len(rows)}] {item['id']}: "
            + ", ".join(f"{n} {r['scores']['error_type']}" for n, r in scored.items()),
            file=sys.stderr,
        )
    return {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "label": label,
        "timestamp": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "config": {
            "llm_model": _model_name(generator),
            **judge_info,
            "judge_votes": JUDGE_VOTES,
        },
        "summary": summarize(rows),
        "skipped": skipped,
        "rows": rows,
        "answers": answers,
    }


def _print_summary(record: dict) -> None:
    s = record["summary"]
    for key in ("passed", "unsupported_answers", "wrong_refusals"):
        print(f"{key}: previous {s[key]['previous']}, revised {s[key]['revised']}")
    c = s["clean_refusals"]
    print(
        f"clean refusals of {c['of']}: previous {c['previous']}, revised {c['revised']}"
    )
    v = s["vital_recall"]
    print(
        f"vital recall on {v['questions']} answerable questions: previous better "
        f"{v['previous_better']}, revised better {v['revised_better']}, ties {v['ties']}"
    )
    for condition, met in s["rule"].items():
        print(f"  {'met' if met else 'NOT met'}: {condition}")
    print(f"adoption rule {'met' if s['rule_met'] else 'not met'}")
    skipped = {k: v for k, v in record["skipped"].items() if v}
    if skipped:
        print(f"skipped: {skipped}")


def main() -> int:
    from dotenv import load_dotenv

    from eval.graph_run import load_outputs
    from eval.history import git_info
    from eval.run_eval import _delay_sec

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outputs",
        type=Path,
        required=True,
        help="Saved outputs of a full run (eval/reports/outputs_<run_id>.json)",
    )
    parser.add_argument(
        "--golden", type=Path, default=_PROJECT_ROOT / "eval" / "data" / "golden.json"
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()

    if git_info().get("dirty"):
        print(
            "Tracked files have uncommitted changes, so the commit would not say "
            "which prompt was tested. Commit first.",
            file=sys.stderr,
        )
        return 1
    golden = json.loads(args.golden.read_text(encoding="utf-8"))[: args.limit]
    outputs = load_outputs(args.outputs)
    record = run(
        golden, outputs, args.label, _delay_sec("EVAL_DELAY_AFTER_GRAPH_SEC", 5.0)
    )
    record["config"]["contexts_from"] = args.outputs.name
    answers = record.pop("answers")
    args.reports.mkdir(parents=True, exist_ok=True)
    report = args.reports / f"prompt_test_{record['run_id']}.json"
    report.write_text(
        json.dumps({**record, "answers": answers}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    with args.log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _print_summary(record)
    print(f"Recorded: {args.log}; answers: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
