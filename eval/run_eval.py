"""Run evaluation (scoring v2): run the graph per golden item, judge, score, report.

Per question:
1. run the graph, capturing the first retrieval and the grade_documents verdict
   (eval/graph_run.py);
2. ask the judge two narrow questions (eval/judge.py);
3. derive all metrics and the error type deterministically (eval/scoring.py).

Full graph outputs are saved to <output-dir>/outputs_<run_id>.json so a run
can be re-scored without re-running the graph.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from eval.dashboard import write_dashboard
from eval.golden import GoldenError, golden_warnings, load_golden
from eval.graph_run import save_outputs
from eval.history import DEFAULT_HISTORY_PATH, append_run, build_run_record, git_info
from eval.judge import judge_item
from eval.scoring import METRICS_VERSION, score_item

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MAX_RATE_LIMIT_RETRIES = 4
_INITIAL_BACKOFF_SEC = 30

# Default delays for eval to stay under LLM rate limits (Mistral free tier ~1 RPS, ~30 RPM)
_DEFAULT_DELAY_AFTER_GRAPH_SEC = (
    5.0  # after graph.invoke, before judging (2+ LLM calls)
)
_DEFAULT_DELAY_BETWEEN_ITEMS_SEC = 20.0  # between items to stay under RPM

# Per-question scores that go into the report, history and dashboard. None
# (not applicable, e.g. vital recall of a refusal question) is left out.
NUMERIC_METRICS = (
    "evidence_recall_initial",
    "evidence_recall_context",
    "vital_recall",
    "vital_recall_lenient",
    "all_recall",
    "grounded_vital_recall",
    "context_utilization",
    "unsupported_claim",  # 1 if the answer has any unsupported claim: lower is better
    "grade_documents_correct",
)


def _is_rate_limit_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        "429" in str(e)
        or "rate limit" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
    )


def _delay_sec(env_name: str, default: float) -> float:
    """Parse delay from env; return default if unset or invalid. Used for eval rate-limit spacing."""
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return max(0.0, v)
    except ValueError:
        return default


def _invoke_with_retry(fn, *args, **kwargs):
    """Call fn; on 429 / rate limit, back off and retry up to _MAX_RATE_LIMIT_RETRIES."""
    last_error = None
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e) and attempt < _MAX_RATE_LIMIT_RETRIES - 1:
                wait = _INITIAL_BACKOFF_SEC * (2**attempt)
                time.sleep(wait)
                continue
            raise
    raise last_error


def _invoke_graph(question: str, graph, llm) -> dict:
    """Run graph for one question; final answer plus first retrieval, sufficiency and node path."""
    from eval.graph_run import run_graph
    from graph.llm_factory import get_embedding_provider

    config = {"run_name": "eval-run", "tags": ["eval", "golden"]}
    return run_graph(
        question,
        graph,
        llm=llm,
        embedding_provider=get_embedding_provider(),
        config=config,
    )


def _model_name(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "model_name", None) or "n/a"


def _judge_llm(graph_llm, llm_provider: str) -> tuple[object, dict]:
    """The judge model and how it was chosen.

    EVAL_GRADER_PROVIDER picks the provider, EVAL_JUDGE_MODEL the model. With
    neither set the answering model judges itself; that is recorded and warned
    about, because a model grading its own answers is lenient.
    """
    from graph.llm_factory import get_llm, scaleway_chat

    provider = (os.getenv("EVAL_GRADER_PROVIDER") or "").strip().lower()
    model = (os.getenv("EVAL_JUDGE_MODEL") or "").strip() or None
    if not provider and not model:
        judge = graph_llm
    elif provider == "scaleway":
        if not model:
            raise ValueError("Set EVAL_JUDGE_MODEL to a Scaleway model id")
        judge = scaleway_chat(model)
    else:
        judge = get_llm(provider=provider or llm_provider, model_variant=model)
    judge_provider = provider or llm_provider
    info = {
        "judge_provider": judge_provider,
        "judge_model": _model_name(judge),
        "judge_is_generator": judge_provider == llm_provider
        and _model_name(judge) == _model_name(graph_llm),
    }
    return judge, info


def score_outputs(
    golden: list[dict], outputs: dict[str, dict], judge_llm
) -> list[dict]:
    """Judge and score every golden item from its saved graph output."""
    results = []
    for item in golden:
        run = outputs[item["id"]]
        verdict = _invoke_with_retry(
            judge_item,
            item,
            run["generation"],
            run.get("context_used_for_generation") or "",
            judge_llm,
        )
        scores = score_item(
            item,
            {
                "initial_documents": run.get("initial_documents") or [],
                "context": run.get("context_used_for_generation") or "",
                "sufficient": run.get("sufficient"),
            },
            verdict,
        )
        results.append(_result(item, run, scores))
    return results


def _result(item: dict, run: dict, scores: dict) -> dict:
    generation = run.get("generation", "")
    return {
        "id": item["id"],
        "question": item["question"],
        "topics": item.get("topics") or [],
        "language": item.get("language"),
        "source": item.get("source"),
        "expected_behavior": item["expected_behavior"],
        "pass": scores["pass"],
        "error_type": scores["error_type"],
        "refused": scores["refused"],
        "unsupported_claims": scores["unsupported_claims"],
        "metrics": _metrics(scores),
        "generation_preview": (
            (generation[:300] + "…") if len(generation) > 300 else generation
        ),
        "retrieval_warning": run.get("retrieval_warning"),
        "web_search_attempted": run.get("web_search_attempted", False),
        "node_path": run.get("node_path") or [],
    }


def _metrics(scores: dict) -> dict[str, float]:
    """Per-question scores on a 0-1 scale; metrics that do not apply are left out."""
    values = dict(scores)
    if values.get("unsupported_claims") is not None:
        values["unsupported_claim"] = values["unsupported_claims"] > 0
    return {
        name: float(values[name])
        for name in NUMERIC_METRICS
        if values.get(name) is not None
    }


def summarize(results: list[dict]) -> dict:
    """Pass rate over judged questions, mean of each metric where it applies,
    and how many questions failed for which reason."""
    judged = [r for r in results if r["pass"] is not None]
    summary = {
        "n": len(results),
        "judged": len(judged),
        "pass_rate": (
            sum(1 for r in judged if r["pass"]) / len(judged) if judged else None
        ),
        "pass_rule": "v2-strict",
        "error_counts": dict(Counter(r["error_type"] for r in results)),
    }
    for name in NUMERIC_METRICS:
        values = [r["metrics"][name] for r in results if name in r["metrics"]]
        summary[f"{name}_mean"] = sum(values) / len(values) if values else None
    return summary


def _run_eval(
    golden: list[dict],
    no_web_search: bool,
    output_dir: Path,
    run_id: str,
    delay_after_graph_sec: float = 0.0,
    delay_between_items_sec: float = 0.0,
) -> tuple[dict, list[dict], dict]:
    """Run the graph for each item, save outputs, judge and score.

    Returns summary, per-question results, and the run header (dataset
    fingerprint + config) that says what was tested.
    """
    if no_web_search:
        os.environ["WEB_SEARCH_ENABLED"] = "false"

    from eval.history import dataset_fingerprint, prompt_fingerprints
    from graph.consts import env_bool
    from graph.graph import app as graph
    from graph.llm_factory import (
        get_embedding_model_name,
        get_embedding_provider,
        get_llm,
    )
    from ingestion import RETRIEVER_K

    graph_llm = get_llm()
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    judge_llm, judge_info = _judge_llm(graph_llm, llm_provider)
    embedding_provider = get_embedding_provider()
    header = {
        "dataset": dataset_fingerprint(golden),
        "config": {
            "llm_provider": llm_provider,
            "llm_model": _model_name(graph_llm),
            **judge_info,
            "embedding_provider": embedding_provider,
            "embedding_model": get_embedding_model_name(embedding_provider),
            "retriever_k": RETRIEVER_K,
            "web_search": env_bool("WEB_SEARCH_ENABLED"),
            "metrics_version": METRICS_VERSION,
            "prompts": prompt_fingerprints(),
        },
    }
    print(
        f"Eval: answer model = {header['config']['llm_model']}, "
        f"judge = {judge_info['judge_provider']}/{judge_info['judge_model']}",
        file=sys.stderr,
    )
    if judge_info["judge_is_generator"]:
        print(
            "Warning: the judge is the answering model (set EVAL_GRADER_PROVIDER / "
            "EVAL_JUDGE_MODEL); self-judged scores tend to be lenient.",
            file=sys.stderr,
        )

    outputs: dict[str, dict] = {}
    for item in golden:
        outputs[item["id"]] = _invoke_with_retry(
            _invoke_graph, item["question"], graph, graph_llm
        )
        if delay_after_graph_sec > 0:
            time.sleep(delay_after_graph_sec)
        if delay_between_items_sec > 0 and item is not golden[-1]:
            time.sleep(delay_between_items_sec)
    save_outputs(outputs, output_dir / f"outputs_{run_id}.json")

    results = score_outputs(golden, outputs, judge_llm)
    return summarize(results), results, header


def _write_report(
    summary: dict,
    results: list[dict],
    output_dir: Path,
    run_id: str,
    header: dict | None = None,
) -> tuple[Path, Path]:
    """Write report_<run_id>.json and report_<run_id>.md; return both paths.

    header: label, notes, git, dataset, config, duration_sec, stored alongside
    the results so a report says what it tested.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"report_{run_id}.json"
    md_path = output_dir / f"report_{run_id}.md"
    header = header or {}

    payload = {"run_id": run_id, **header, "summary": summary, "results": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    pass_rate = summary["pass_rate"]
    lines = [
        "# Evaluation report",
        "",
        f"**Run ID:** {run_id}",
        "",
        *_markdown_header_lines(header),
        "## Summary",
        "",
        (
            f"- **Pass rate:** {pass_rate:.0%} of {summary['judged']} judged questions"
            if pass_rate is not None
            else "- **Pass rate:** no question could be judged"
        ),
        "- **Outcomes:** "
        + ", ".join(f"{k} {v}" for k, v in sorted(summary["error_counts"].items())),
        *[
            f"- **{name}:** {summary[f'{name}_mean']:.2f}"
            for name in NUMERIC_METRICS
            if summary.get(f"{name}_mean") is not None
        ],
        "",
        "## Per-question results",
        "",
    ]
    for r in results:
        status = {True: "PASS", False: "FAIL", None: "NOT JUDGED"}[r["pass"]]
        lines += [
            f"### {r['id']} — {status} ({r['error_type']})",
            "",
            f"- **Question:** {r['question'][:200]}{'…' if len(r['question']) > 200 else ''}",
            "- **Scores:** "
            + ", ".join(f"{k}={v:.2f}" for k, v in r["metrics"].items()),
            f"- **Path:** {' → '.join(r['node_path'])}",
            f"- **Answer (preview):** {r['generation_preview'][:150]}…",
        ]
        if r.get("retrieval_warning"):
            lines.append(f"- **Warning:** {r['retrieval_warning']}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def _markdown_header_lines(header: dict) -> list[str]:
    """What the run tested, for the top of the Markdown report."""
    if not header:
        return []
    git = header.get("git") or {}
    config = header.get("config") or {}
    dataset = header.get("dataset") or {}
    commit = git.get("commit") or "unknown"
    if git.get("dirty"):
        commit += " (uncommitted changes)"
    lines = []
    if header.get("label"):
        lines.append(f"**Label:** {header['label']}")
    if header.get("notes"):
        lines.append(f"**Notes:** {header['notes']}")
    lines += [
        f"**Commit:** {commit} on {git.get('branch') or 'unknown'}",
        f"**Models:** {config.get('llm_model')} (judge: "
        f"{config.get('judge_provider')}/{config.get('judge_model')}, "
        f"embeddings: {config.get('embedding_model')})",
        f"**Retrieval:** k={config.get('retriever_k')} per collection,"
        f" web search {'on' if config.get('web_search') else 'off'}",
        f"**Questions:** {dataset.get('n_items')} (set {dataset.get('questions_hash')})",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation against golden dataset."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=_PROJECT_ROOT / "eval" / "data" / "golden.json",
        help="Path to golden JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only on first N items",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search for reproducible eval",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "eval" / "reports",
        help="Directory for report outputs",
    )
    parser.add_argument(
        "--delay-after-graph",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to wait after each graph run (env: EVAL_DELAY_AFTER_GRAPH_SEC; default 5)",
    )
    parser.add_argument(
        "--delay-between-items",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to wait between golden items (env: EVAL_DELAY_BETWEEN_ITEMS_SEC; default 20)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Short name for this run in the history and dashboard (e.g. dk-query-translation)",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Free-text notes on what this run tests",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Run history to append to (default: eval/history/runs.jsonl)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not record this run in the history",
    )
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}", file=sys.stderr)
        return 1
    try:
        golden = load_golden(args.golden)
    except GoldenError as err:
        print(err, file=sys.stderr)
        return 1
    for warning in golden_warnings(golden):
        print(f"Warning: {warning}", file=sys.stderr)
    if args.limit is not None:
        golden = golden[: args.limit]

    delay_after_graph = (
        args.delay_after_graph
        if args.delay_after_graph is not None
        else _delay_sec("EVAL_DELAY_AFTER_GRAPH_SEC", _DEFAULT_DELAY_AFTER_GRAPH_SEC)
    )
    delay_between_items = (
        args.delay_between_items
        if args.delay_between_items is not None
        else _delay_sec(
            "EVAL_DELAY_BETWEEN_ITEMS_SEC", _DEFAULT_DELAY_BETWEEN_ITEMS_SEC
        )
    )
    started = time.monotonic()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    # Before the run: describes the code that ran, not edits made while it ran.
    git = git_info()
    summary, results, header = _run_eval(
        golden,
        no_web_search=args.no_web_search,
        output_dir=args.output_dir,
        run_id=run_id,
        delay_after_graph_sec=delay_after_graph,
        delay_between_items_sec=delay_between_items,
    )
    header = {
        "label": args.label,
        "notes": args.notes,
        "git": git,
        **header,
        "duration_sec": round(time.monotonic() - started, 1),
    }
    json_path, md_path = _write_report(
        summary, results, args.output_dir, run_id, header
    )
    if not args.no_history:
        append_run(
            build_run_record(
                run_id=run_id,
                summary=summary,
                results=results,
                report_file=json_path.name,
                **header,
            ),
            args.history_file,
        )
        print(f"Run recorded: {args.history_file}")
        dashboard = write_dashboard(
            args.output_dir / "dashboard.html",
            args.history_file,
            reports_dir=args.output_dir,
        )
        print(f"Dashboard updated: {dashboard}")
    print(f"Report written: {json_path}")
    print(f"Report written: {md_path}")
    if summary["pass_rate"] is not None:
        print(f"Pass rate: {summary['pass_rate']:.0%} of {summary['judged']} judged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
