"""Run evaluation: load golden dataset, invoke graph, compute metrics, write report."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root for default paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _invoke_graph(question: str, graph, llm) -> dict:
    """Run graph for one question; return state slice we need for metrics and report."""
    invoke_input = {
        "question": question,
        "generation": "",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
        "chat_history": [],
        "llm": llm,
    }
    config = {"run_name": "eval-run", "tags": ["eval", "golden"]}
    result = graph.invoke(invoke_input, config=config)
    return {
        "generation": result.get("generation", ""),
        "documents": result.get("documents", []),
        "retrieval_warning": result.get("retrieval_warning"),
        "web_search_attempted": result.get("web_search_attempted", False),
    }


def _run_eval(
    golden_path: Path,
    limit: int | None,
    no_web_search: bool,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    """Load golden, run graph and metrics, return summary and results for reporting."""
    golden = _load_golden(golden_path)
    if limit is not None:
        golden = golden[:limit]

    if no_web_search:
        os.environ["WEB_SEARCH_ENABLED"] = "false"

    from graph.graph import app as graph
    from graph.llm_factory import get_llm

    from eval.metrics import compute_all_metrics

    llm = get_llm()
    results = []
    for item in golden:
        question = item["question"]
        item_id = item.get("id", "")
        expected_answer = item.get("expected_answer")
        key_facts = item.get("key_facts")
        run = _invoke_graph(question, graph, llm)
        generation = run["generation"]
        documents = run["documents"]
        metrics = compute_all_metrics(
            question=question,
            generation=generation,
            documents=documents,
            expected_answer=expected_answer,
            key_facts=key_facts,
            llm=llm,
        )
        threshold = 0.5
        passed = all(m >= threshold for m in metrics.values())
        results.append({
            "id": item_id,
            "question": question,
            "pass": passed,
            "metrics": metrics,
            "generation_preview": (generation[:300] + "…") if len(generation) > 300 else generation,
            "retrieval_warning": run["retrieval_warning"],
            "web_search_attempted": run["web_search_attempted"],
        })

    n = len(results)
    summary = {
        "pass_rate": sum(1 for r in results if r["pass"]) / n if n else 0.0,
        "faithfulness_mean": sum(r["metrics"]["faithfulness"] for r in results) / n if n else 0.0,
        "answer_relevance_mean": sum(r["metrics"]["answer_relevance"] for r in results) / n if n else 0.0,
        "context_precision_mean": sum(r["metrics"]["context_precision"] for r in results) / n if n else 0.0,
        "context_recall_mean": sum(r["metrics"]["context_recall"] for r in results) / n if n else 0.0,
    }
    return summary, results


def _write_report(summary: dict, results: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Write report_<timestamp>.json and report_<timestamp>.md; return both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"report_{ts}.json"
    md_path = output_dir / f"report_{ts}.md"

    payload = {"summary": summary, "results": results, "run_id": ts}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# Evaluation report",
        "",
        f"**Run ID:** {ts}",
        "",
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
        lines.append(f"- **Question:** {r['question'][:200]}{'…' if len(r['question']) > 200 else ''}")
        lines.append(f"- **Metrics:** faithfulness={r['metrics']['faithfulness']:.2f}, answer_relevance={r['metrics']['answer_relevance']:.2f}, context_precision={r['metrics']['context_precision']:.2f}, context_recall={r['metrics']['context_recall']:.2f}")
        lines.append(f"- **Generation (preview):** {r['generation_preview'][:150]}…")
        if r.get("retrieval_warning"):
            lines.append(f"- **Warning:** {r['retrieval_warning']}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation against golden dataset.")
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
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}", file=sys.stderr)
        return 1
    summary, results = _run_eval(
        golden_path=args.golden,
        limit=args.limit,
        no_web_search=args.no_web_search,
        output_dir=args.output_dir,
    )
    json_path, md_path = _write_report(summary, results, args.output_dir)
    print(f"Report written: {json_path}")
    print(f"Report written: {md_path}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
