"""Pooling: retrieved chunks that no evidence quote covers yet, for human review.

A retrieval variant can find a valid passage that the golden set does not list
as evidence; it then scores a miss, and comparisons favour the retriever the
labels were made with (BEIR, Thakur et al. 2021). This lists, per question,
the chunks in the top k of saved retrieval-only runs that contain no quote.
A person reads them and adds confirmed passages as further evidence quotes;
`run_eval --rescore RUN_ID` then scores every run again against the same labels.

    uv run python -m eval.pool RUN_ID [RUN_ID ...] [--k 5] [--all]

LLM judgements are deliberately not used to decide relevance: their errors
lean towards missed relevant passages (Thomas et al. 2024), and a label they
reject would silently stay a miss.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval.golden import GoldenError, load_golden
from eval.graph_run import load_outputs
from eval.scoring import evidence_found

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COLLECTIONS = ("iaea", "dk")


def pool_candidates(item: dict, runs: dict[str, dict[str, list]], k: int) -> list[dict]:
    """Chunks in the top k (per collection) of any run that hold no quote of
    any nugget, each listed once with the rank at which each run found it."""
    quotes = [q for n in item.get("nuggets") or [] for q in n["evidence"]]
    candidates: dict[str, dict] = {}
    for run_id, ranked in runs.items():
        for collection in _COLLECTIONS:
            for rank, doc in enumerate(ranked.get(collection, [])[:k], start=1):
                if evidence_found(quotes, [doc.page_content]):
                    continue
                entry = candidates.setdefault(
                    doc.page_content,
                    {
                        "text": doc.page_content,
                        "source": (doc.metadata or {}).get("source"),
                        "collection": collection,
                        "found_by": {},
                    },
                )
                entry["found_by"][run_id] = rank
    return list(candidates.values())


def _missed(item: dict, ranked: dict[str, list], k: int) -> bool:
    """Whether some vital nugget has no quote in this run's top k."""
    texts = [d.page_content for c in _COLLECTIONS for d in ranked.get(c, [])[:k]]
    return any(
        not evidence_found(n["evidence"], texts)
        for n in item.get("nuggets") or []
        if n["importance"] == "vital"
    )


def _report(golden: list[dict], runs: dict[str, dict], k: int, all_items: bool) -> str:
    lines = [
        "# Evidence pool for review",
        "",
        f"Runs: {', '.join(runs)} · top {k} per collection",
        "",
        "For each chunk: if it states a nugget's fact, copy the shortest verbatim "
        "span that tells this passage apart into that nugget's `evidence` list.",
        "",
    ]
    for item in golden:
        per_run = {
            rid: run[item["id"]] for rid, run in runs.items() if item["id"] in run
        }
        if not item.get("nuggets") or not per_run:
            continue
        if not all_items and not any(_missed(item, r, k) for r in per_run.values()):
            continue
        candidates = pool_candidates(item, per_run, k)
        lines += [f"## {item['id']}", "", f"**Question:** {item['question']}", ""]
        lines += [
            f"- {n['importance']}: {n['text']} (quotes: "
            + "; ".join(repr(q) for q in n["evidence"])
            + ")"
            for n in item["nuggets"]
        ]
        lines.append("")
        for c in candidates:
            ranks = ", ".join(f"{rid} #{rank}" for rid, rank in c["found_by"].items())
            lines += [
                f"### {c['source'] or 'unknown source'} ({c['collection']}; {ranks})",
                "",
                "> " + c["text"].replace("\n", "\n> "),
                "",
            ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    parser.add_argument(
        "--golden", type=Path, default=_PROJECT_ROOT / "eval" / "data" / "golden.json"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=_PROJECT_ROOT / "eval" / "reports"
    )
    parser.add_argument("--k", type=int, default=5, help="Top k per collection")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also list questions where every run found the evidence",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        golden = load_golden(args.golden)
    except GoldenError as err:
        print(err, file=sys.stderr)
        return 1
    runs = {}
    for run_id in args.run_ids:
        path = args.reports_dir / f"outputs_{run_id}.json"
        if not path.exists():
            print(f"There are no saved outputs for run {run_id}", file=sys.stderr)
            return 1
        outputs = load_outputs(path)
        runs[run_id] = {
            item_id: {
                "iaea": o.get("ranked_iaea") or [],
                "dk": o.get("ranked_dk") or [],
            }
            for item_id, o in outputs.items()
            if "ranked_dk" in o
        }
    output = args.output or args.reports_dir / (
        f"pool_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_report(golden, runs, args.k, args.all), encoding="utf-8")
    print(f"Pool written: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
