"""Do more chunks per collection give better answers? (step 5)

Every golden question is answered twice: from the top 3 chunks of each
collection (what the graph retrieves today) and from the top 5. Retrieval is
deterministic, so the two contexts differ only by the extra chunks. The judge
labels both answers; questions are grouped by where their vital evidence is:

- evidence at k=3: the extra chunks can only distract;
- evidence only at k=5: the extra chunks are what can help;
- evidence missing at k=5: neither context holds it;
- should refuse: more context can tempt the generator to answer anyway.

Recall at 5 above recall at 3 is not enough by itself: answer accuracy
saturates while recall keeps rising (Lost in the Middle §5), so k is judged by
answer quality per group.

    uv run python -m eval.k_test --label k3-vs-k5
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from eval.position_test import _answer_and_judge, compare
from eval.scoring import evidence_found

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = _PROJECT_ROOT / "eval" / "history" / "k_tests.jsonl"
SMALL, LARGE = 3, 5


def contexts(
    iaea: list[Document], dk: list[Document], ks: tuple[int, ...]
) -> dict[int, list[Document]]:
    """For each k, the top k of each collection merged the way the graph does."""
    from graph.nodes.retrieval_common import merge_unique_documents

    return {k: merge_unique_documents([], iaea[:k] + dk[:k])[0] for k in ks}


def _all_vital_evidenced(item: dict, documents: list[Document]) -> bool:
    texts = [d.page_content for d in documents]
    return all(
        evidence_found(n["evidence"], texts)
        for n in item["nuggets"]
        if n["importance"] == "vital"
    )


def stratum(item: dict, small: list[Document], large: list[Document]) -> str:
    if item["expected_behavior"] != "answer":
        return "should refuse"
    if _all_vital_evidenced(item, small):
        return "evidence at k=3"
    if _all_vital_evidenced(item, large):
        return "evidence only at k=5"
    return "evidence missing at k=5"


def run(golden: list[dict], label: str | None, delay_sec: float) -> dict:
    from eval.history import git_info
    from eval.run_eval import _judge_llm, _model_name, _retrieve_ranked
    from graph.llm_factory import DEFAULT_PROVIDER, get_embedding_provider, get_llm

    generator = get_llm()
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    judge, judge_info = _judge_llm(generator, provider, _model_name(generator))
    embedding_provider = get_embedding_provider()

    rows, judge_errors = [], []
    for n, item in enumerate(golden, start=1):
        ranked = _retrieve_ranked(item["question"], embedding_provider, LARGE)
        by_k = contexts(ranked["iaea"], ranked["dk"], (SMALL, LARGE))
        results = {
            k: _answer_and_judge(item, docs, generator, judge, delay_sec)
            for k, docs in by_k.items()
        }
        if any(r["score"] is None for r in results.values()):
            judge_errors.append(item["id"])
            continue
        rows.append(
            {
                "id": item["id"],
                "language": item.get("language"),
                "stratum": stratum(item, by_k[SMALL], by_k[LARGE]),
                "k3": results[SMALL]["score"],
                "k5": results[LARGE]["score"],
                "unsupported_k3": results[SMALL]["unsupported"],
                "unsupported_k5": results[LARGE]["unsupported"],
                "refused_k3": results[SMALL]["refused"],
                "refused_k5": results[LARGE]["refused"],
            }
        )
        print(
            f"  [{n}/{len(golden)}] {item['id']} ({rows[-1]['stratum']}): "
            f"k3 {rows[-1]['k3']:.2f}, k5 {rows[-1]['k5']:.2f}",
            file=sys.stderr,
        )
    strata = sorted({r["stratum"] for r in rows})
    return {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "label": label,
        "timestamp": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "config": {
            "llm_model": _model_name(generator),
            **judge_info,
            "embedding_provider": embedding_provider,
            "judge_votes": 1,
            "k": [SMALL, LARGE],
        },
        "summary": {
            "all": compare(rows, "k3", "k5"),
            **{
                s: compare([r for r in rows if r["stratum"] == s], "k3", "k5")
                for s in strata
            },
            "unsupported_answers": {
                "k3": sum(r["unsupported_k3"] > 0 for r in rows),
                "k5": sum(r["unsupported_k5"] > 0 for r in rows),
            },
        },
        "judge_errors": judge_errors,
        "rows": rows,
    }


def _print_summary(record: dict) -> None:
    for name, s in record["summary"].items():
        if name == "unsupported_answers":
            print(f"answers with an unsupported claim: k3 {s['k3']}, k5 {s['k5']}")
            continue
        p = s["sign_test"]["p_value"]
        print(
            f"{name}: {s['questions']} questions, k3 better {s['k3_better']}, "
            f"k5 better {s['k5_better']}, ties {s['ties']}"
            + ("" if p is None else f" (sign test p = {p:.3f})")
        )
    if record["judge_errors"]:
        print(f"judge errors: {record['judge_errors']}")


def main() -> int:
    from dotenv import load_dotenv

    from eval.run_eval import _delay_sec

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--golden", type=Path, default=_PROJECT_ROOT / "eval" / "data" / "golden.json"
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))[: args.limit]
    record = run(golden, args.label, _delay_sec("EVAL_DELAY_AFTER_GRAPH_SEC", 5.0))
    with args.log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _print_summary(record)
    print(f"Recorded: {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
