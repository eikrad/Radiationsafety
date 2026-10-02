"""Does the generator use evidence less when it sits in the middle of the context? (#132)

For every answerable golden question whose first retrieval holds the evidence
for a vital nugget, the generator answers twice from the same chunks: once
with the evidence chunks first, once with them in the middle, after half of
the other chunks. With 3 IAEA and 3 Danish chunks that middle is position 4,
where the best Danish chunk sits today. The judge labels the vital nuggets of
both answers; questions where the two scores differ go into a sign test.

Lost in the Middle (Liu et al. 2024) found answers 20 points worse with the
evidence in the middle of 20 documents for 2023 models, much less for others,
and never tested 6 long chunks; so the effect is measured, not assumed.

    uv run python -m eval.position_test --label position-k3
"""

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from eval.dashboard import sign_test
from eval.scoring import _credit, evidence_found

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = _PROJECT_ROOT / "eval" / "history" / "position_tests.jsonl"


def evidence_indices(item: dict, documents: list[Document]) -> list[int]:
    """Positions of the chunks that hold the evidence for a vital nugget."""
    quotes = [
        q
        for n in item.get("nuggets") or []
        if n["importance"] == "vital"
        for q in n["evidence"]
    ]
    return [
        i for i, d in enumerate(documents) if evidence_found(quotes, [d.page_content])
    ]


def placements(documents: list[Document], evidence: list[int]) -> dict[str, list]:
    """The same chunks with the evidence first, or after half of the others
    (rounded up); evidence and other chunks each keep their retrieval order."""
    chosen = [documents[i] for i in evidence]
    others = [d for i, d in enumerate(documents) if i not in evidence]
    half = math.ceil(len(others) / 2)
    return {
        "first": chosen + others,
        "middle": others[:half] + chosen + others[half:],
    }


def vital_score(item: dict, labels: list[str]) -> float:
    """Lenient vital recall: support 1, partial support ½, per vital nugget."""
    vital = [i for i, n in enumerate(item["nuggets"]) if n["importance"] == "vital"]
    return sum(_credit(labels[i]) for i in vital) / len(vital)


def compare(rows: list[dict]) -> dict:
    """Paired outcome: how often each placement scored higher, and a sign test
    over the questions where the two differ."""
    first_better = sum(r["first"] > r["middle"] for r in rows)
    middle_better = sum(r["middle"] > r["first"] for r in rows)
    n = len(rows)
    return {
        "questions": n,
        "first_better": first_better,
        "middle_better": middle_better,
        "ties": n - first_better - middle_better,
        "mean_first": sum(r["first"] for r in rows) / n if n else None,
        "mean_middle": sum(r["middle"] for r in rows) / n if n else None,
        "sign_test": sign_test(first_better, middle_better),
    }


def _answer_and_judge(item, documents, generator, judge, delay_sec: float) -> dict:
    from eval.judge import judge_item
    from eval.run_eval import _invoke_with_retry
    from graph.chains.generation import get_generation_chain
    from graph.chains.truncate import format_context

    context = format_context(documents)
    answer = _invoke_with_retry(
        get_generation_chain(generator).invoke,
        {"context": context, "chat_history": "", "question": item["question"]},
    )
    time.sleep(delay_sec)
    verdict = judge_item(item, answer, context, judge, votes=1)
    time.sleep(delay_sec)
    if verdict is None:
        return {"score": None, "unsupported": None, "answer": answer}
    return {
        "score": vital_score(item, verdict["nuggets"]),
        "unsupported": len(verdict["unsupported_claims"]),
        "answer": answer,
    }


def run(golden: list[dict], label: str | None, delay_sec: float) -> dict:
    import os

    from eval.history import git_info
    from eval.run_eval import _judge_llm, _model_name, _retrieve_initial
    from graph.llm_factory import DEFAULT_PROVIDER, get_embedding_provider, get_llm

    generator = get_llm()
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    judge, judge_info = _judge_llm(generator, provider, _model_name(generator))
    embedding_provider = get_embedding_provider()

    rows, skipped = [], {"no_evidence": [], "only_evidence": [], "judge_error": []}
    answerable = [i for i in golden if i["expected_behavior"] == "answer"]
    for n, item in enumerate(answerable, start=1):
        documents = _retrieve_initial(item["question"], embedding_provider)
        evidence = evidence_indices(item, documents)
        if not evidence:
            skipped["no_evidence"].append(item["id"])
            continue
        if len(evidence) == len(documents):
            skipped["only_evidence"].append(item["id"])
            continue
        results = {
            name: _answer_and_judge(item, docs, generator, judge, delay_sec)
            for name, docs in placements(documents, evidence).items()
        }
        if any(r["score"] is None for r in results.values()):
            skipped["judge_error"].append(item["id"])
            continue
        rows.append(
            {
                "id": item["id"],
                "language": item.get("language"),
                "chunks": len(documents),
                "evidence_at": [i + 1 for i in evidence],
                "first": results["first"]["score"],
                "middle": results["middle"]["score"],
                "unsupported_first": results["first"]["unsupported"],
                "unsupported_middle": results["middle"]["unsupported"],
            }
        )
        print(
            f"  [{n}/{len(answerable)}] {item['id']}: first {rows[-1]['first']:.2f}, "
            f"middle {rows[-1]['middle']:.2f}",
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
            "embedding_provider": embedding_provider,
            "judge_votes": 1,
        },
        "summary": compare(rows),
        "skipped": skipped,
        "rows": rows,
    }


def _print_summary(record: dict) -> None:
    s = record["summary"]
    sig = s["sign_test"]
    p = "n/a" if sig["p_value"] is None else f"{sig['p_value']:.3f}"
    print(
        f"{s['questions']} questions with evidence: first better {s['first_better']}, "
        f"middle better {s['middle_better']}, ties {s['ties']} "
        f"(sign test p = {p}); mean vital recall first "
        f"{s['mean_first'] or 0:.2f}, middle {s['mean_middle'] or 0:.2f}"
    )
    skipped = {k: len(v) for k, v in record["skipped"].items() if v}
    if skipped:
        print(f"skipped: {skipped}")


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
    delay = _delay_sec("EVAL_DELAY_AFTER_GRAPH_SEC", 5.0)
    record = run(golden, args.label, delay)
    with args.log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _print_summary(record)
    print(f"Recorded: {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
