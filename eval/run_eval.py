"""Run evaluation: load golden dataset, invoke graph, compute metrics, write report."""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from eval.dashboard import write_dashboard
from eval.history import DEFAULT_HISTORY_PATH, append_run, build_run_record, git_info

# Project root for default paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CACHE_FILENAME = "eval_cache.json"

_MAX_RATE_LIMIT_RETRIES = 4
_INITIAL_BACKOFF_SEC = 30

# Default delays for eval to stay under LLM rate limits (Mistral free tier ~1 RPS, ~30 RPM)
_DEFAULT_DELAY_AFTER_GRAPH_SEC = (
    5.0  # after graph.invoke, before metrics (4+ LLM calls)
)
_DEFAULT_DELAY_BETWEEN_ITEMS_SEC = 20.0  # between items to stay under RPM


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


def _load_golden(path: Path) -> list[dict]:
    """Load and validate golden JSON. Each item must have 'question'; optional id, expected_answer, key_facts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Golden file must be a JSON array")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"Item {i}: must be an object with 'question'")
    return data


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
    """Run graph for one question; return state slice we need for metrics and report."""
    from graph.llm_factory import get_embedding_provider

    invoke_input = {
        "question": question,
        "generation": "",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
        "chat_history": [],
        "llm": llm,
        "embedding_provider": get_embedding_provider(),
    }
    config = {"run_name": "eval-run", "tags": ["eval", "golden"]}
    result = graph.invoke(invoke_input, config=config)
    return {
        "generation": result.get("generation", ""),
        "documents": result.get("documents", []),
        "context_used_for_generation": result.get("context_used_for_generation") or "",
        "retrieval_warning": result.get("retrieval_warning"),
        "web_search_attempted": result.get("web_search_attempted", False),
    }


def _serialize_documents(documents: list) -> list[dict]:
    """Serialize Document list to JSON-serializable list of dicts."""
    out = []
    for d in documents:
        meta = getattr(d, "metadata", None) or {}
        out.append(
            {
                "page_content": getattr(d, "page_content", "") or "",
                "metadata": dict(meta),
            }
        )
    return out


def _deserialize_documents(data: list[dict]) -> list[Document]:
    """Deserialize list of dicts back to Document list."""
    return [
        Document(page_content=x.get("page_content", ""), metadata=x.get("metadata", {}))
        for x in data
    ]


def _run_eval(
    golden_path: Path,
    limit: int | None,
    no_web_search: bool,
    output_dir: Path,
    cache_dir: Path | None,
    use_per_chunk_precision: bool,
    pass_rule: str = "all",
    delay_after_graph_sec: float = 0.0,
    delay_between_items_sec: float = 0.0,
) -> tuple[dict, list[dict], dict]:
    """Load golden, run graph and metrics.

    Returns summary, per-question results, and the run header (dataset
    fingerprint + config) that says what was tested.
    """
    golden = _load_golden(golden_path)
    if limit is not None:
        golden = golden[:limit]

    if no_web_search:
        os.environ["WEB_SEARCH_ENABLED"] = "false"

    from eval import metrics
    from eval.history import dataset_fingerprint, prompt_fingerprints
    from graph.consts import env_bool
    from graph.graph import app as graph
    from graph.llm_factory import (
        get_embedding_model_name,
        get_embedding_provider,
        get_llm,
    )
    from ingestion import RETRIEVER_K

    golden_mtime = golden_path.stat().st_mtime
    cache: dict = {}
    cache_path = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / _CACHE_FILENAME
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                if (
                    data.get("golden_path") == str(golden_path.resolve())
                    and data.get("golden_mtime") == golden_mtime
                ):
                    cache = data.get("entries", {})
            except (json.JSONDecodeError, OSError):
                pass

    # Graph uses main provider; grading can use a different provider via EVAL_GRADER_PROVIDER
    graph_llm = get_llm()
    grader_provider = (os.getenv("EVAL_GRADER_PROVIDER") or "").strip().lower()
    grader_llm = get_llm(provider=grader_provider) if grader_provider else graph_llm
    graph_model = _model_name(graph_llm)
    grader_model = _model_name(grader_llm)
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    embedding_provider = get_embedding_provider()
    header = {
        "dataset": dataset_fingerprint(golden),
        "config": {
            "llm_provider": llm_provider,
            "llm_model": graph_model,
            "grader_provider": grader_provider or llm_provider,
            "grader_model": grader_model,
            "embedding_provider": embedding_provider,
            "embedding_model": get_embedding_model_name(embedding_provider),
            "retriever_k": RETRIEVER_K,
            "web_search": env_bool("WEB_SEARCH_ENABLED"),
            "pass_rule": pass_rule,
            "per_chunk_precision": use_per_chunk_precision,
            "metrics_version": metrics.METRICS_VERSION,
            "cache_used": bool(cache_dir),
            "prompts": prompt_fingerprints(),
        },
    }
    print(
        f"Eval: graph model = {graph_model}, grader model = {grader_model}",
        file=sys.stderr,
    )
    results = []
    for item in golden:
        question = item["question"]
        item_id = item.get("id", "") or str(hash(question))
        expected_answer = item.get("expected_answer")
        key_facts = item.get("key_facts")
        if cache_dir and item_id in cache:
            entry = cache[item_id]
            generation = entry.get("generation", "")
            documents = _deserialize_documents(entry.get("documents", []))
            context_used_for_generation = entry.get("context_used_for_generation") or ""
            retrieval_warning = entry.get("retrieval_warning")
            web_search_attempted = entry.get("web_search_attempted", False)
        else:
            run = _invoke_with_retry(_invoke_graph, question, graph, graph_llm)
            generation = run["generation"]
            documents = run["documents"]
            context_used_for_generation = run.get("context_used_for_generation") or ""
            retrieval_warning = run["retrieval_warning"]
            web_search_attempted = run["web_search_attempted"]
            if cache_dir:
                cache[item_id] = {
                    "generation": generation,
                    "documents": _serialize_documents(documents),
                    "context_used_for_generation": context_used_for_generation,
                    "retrieval_warning": retrieval_warning,
                    "web_search_attempted": web_search_attempted,
                }
        if delay_after_graph_sec > 0:
            time.sleep(delay_after_graph_sec)
        scores = _invoke_with_retry(
            metrics.compute_all_metrics,
            question=question,
            generation=generation,
            documents=documents,
            context_used_for_generation=context_used_for_generation,
            expected_answer=expected_answer,
            key_facts=key_facts,
            llm=grader_llm,
            use_per_chunk_precision=use_per_chunk_precision,
        )
        threshold = 0.5
        if pass_rule == "mean":
            passed = (sum(scores.values()) / len(scores)) >= threshold
        else:
            passed = all(m >= threshold for m in scores.values())
        results.append(
            {
                "id": item.get("id", ""),
                "question": question,
                "topics": item.get("topics") or [],
                "language": item.get("language"),
                "source": item.get("source"),
                "pass": passed,
                "metrics": scores,
                "generation_preview": (
                    (generation[:300] + "…") if len(generation) > 300 else generation
                ),
                "retrieval_warning": retrieval_warning,
                "web_search_attempted": web_search_attempted,
            }
        )
        if delay_between_items_sec > 0 and item is not golden[-1]:
            time.sleep(delay_between_items_sec)

    if cache_dir and cache_path is not None:
        try:
            payload = {
                "golden_path": str(golden_path.resolve()),
                "golden_mtime": golden_mtime,
                "entries": cache,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    n = len(results)
    summary = {
        "pass_rate": sum(1 for r in results if r["pass"]) / n if n else 0.0,
        "pass_rule": pass_rule,
        "faithfulness_mean": (
            sum(r["metrics"]["faithfulness"] for r in results) / n if n else 0.0
        ),
        "answer_relevance_mean": (
            sum(r["metrics"]["answer_relevance"] for r in results) / n if n else 0.0
        ),
        "context_precision_mean": (
            sum(r["metrics"]["context_precision"] for r in results) / n if n else 0.0
        ),
        "context_recall_mean": (
            sum(r["metrics"]["context_recall"] for r in results) / n if n else 0.0
        ),
    }
    return summary, results, header


def _model_name(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "model_name", None) or "n/a"


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
    ts = run_id
    json_path = output_dir / f"report_{ts}.json"
    md_path = output_dir / f"report_{ts}.md"
    header = header or {}

    payload = {"run_id": ts, **header, "summary": summary, "results": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# Evaluation report",
        "",
        f"**Run ID:** {ts}",
        "",
        *_markdown_header_lines(header),
        "## Summary",
        "",
        f"- **Pass rate:** {summary['pass_rate']:.2%}",
        f"- **Faithfulness (mean):** {summary['faithfulness_mean']:.3f}",
        f"- **Answer relevance (mean):** {summary['answer_relevance_mean']:.3f}",
        f"- **Context precision (mean):** {summary['context_precision_mean']:.3f}",
        f"- **Context recall (mean):** {summary['context_recall_mean']:.3f}",
        "",
        "## Per-question results",
        "",
    ]
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        lines.append(f"### {r['id'] or '(no id)'} — {status}")
        lines.append("")
        lines.append(
            f"- **Question:** {r['question'][:200]}{'…' if len(r['question']) > 200 else ''}"
        )
        lines.append(
            f"- **Metrics:** faithfulness={r['metrics']['faithfulness']:.2f}, answer_relevance={r['metrics']['answer_relevance']:.2f}, context_precision={r['metrics']['context_precision']:.2f}, context_recall={r['metrics']['context_recall']:.2f}"
        )
        lines.append(f"- **Generation (preview):** {r['generation_preview'][:150]}…")
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
        f"**Models:** {config.get('llm_model')} (grader: {config.get('grader_model')},"
        f" embeddings: {config.get('embedding_model')})",
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
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for graph outputs (env: EVAL_CACHE_DIR); re-run only metrics when cache hit",
    )
    parser.add_argument(
        "--per-chunk-precision",
        action="store_true",
        help="Use per-chunk context precision (Option A) instead of sufficiency (Option B)",
    )
    parser.add_argument(
        "--pass-rule",
        choices=("all", "mean"),
        default="all",
        help="Pass when all metrics >= 0.5 (all) or mean of metrics >= 0.5 (mean); default: all",
    )
    parser.add_argument(
        "--delay-after-graph",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to wait after graph invoke before running metrics (env: EVAL_DELAY_AFTER_GRAPH_SEC; default 5)",
    )
    parser.add_argument(
        "--delay-between-items",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to wait between processing each golden item (env: EVAL_DELAY_BETWEEN_ITEMS_SEC; default 20)",
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

    cache_dir = args.cache_dir or (
        os.environ.get("EVAL_CACHE_DIR") and Path(os.environ["EVAL_CACHE_DIR"])
    )
    if cache_dir is not None and not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}", file=sys.stderr)
        return 1
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
        golden_path=args.golden,
        limit=args.limit,
        no_web_search=args.no_web_search,
        output_dir=args.output_dir,
        cache_dir=cache_dir,
        use_per_chunk_precision=args.per_chunk_precision,
        pass_rule=args.pass_rule,
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
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
