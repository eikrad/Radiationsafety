"""Golden set format v2: questions with nuggets, evidence quotes and expected behaviour.

    {"id": "...", "question": "...",
     "topics": ["medical"], "language": "da", "source": "dk-law",
     "expected_behavior": "answer" | "refuse",           # default "answer"
     "nuggets": [{"text": "...",                          # atomic fact the answer needs
                  "importance": "vital" | "okay",
                  "evidence": ["verbatim quote", ...]}]}  # any one quote counts

A nugget is judged against the answer by the LLM judge; its evidence quotes are
matched verbatim against retrieved chunks, so they must be copied from the
source document in its own language. Several quotes are alternatives (the same
fact stated in two documents), not parts of one fact.
"""

import json
from pathlib import Path

TOPICS = (
    "medical",
    "industrial",
    "research",
    "transport",
    "waste",
    "emergency",
    "occupational",
    "general",
)
IMPORTANCE = ("vital", "okay")
BEHAVIOURS = ("answer", "refuse")

# A quote longer than this is more likely to straddle two chunks and never match.
MAX_QUOTE_CHARS = 150


class GoldenError(ValueError):
    """The golden file is malformed; the message lists every problem found."""


def load_golden(path: Path) -> list[dict]:
    """Load and validate a v2 golden file; defaults are filled in.

    Raises GoldenError listing all problems, so a broken set is fixed in one pass.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise GoldenError("Golden file must be a JSON array of items")

    items = [_with_defaults(item) for item in data]
    problems = _problems(items)
    if problems:
        raise GoldenError(
            f"{len(problems)} problem(s) in {path}:\n  " + "\n  ".join(problems)
        )
    return items


def golden_warnings(items: list[dict]) -> list[str]:
    """Issues that do not block a run but likely cost evidence matches."""
    warnings = []
    for item in items:
        for n, nugget in enumerate(item["nuggets"], start=1):
            for quote in nugget["evidence"]:
                if len(quote) > MAX_QUOTE_CHARS:
                    warnings.append(
                        f"{item['id']}: {nugget['importance']} nugget {n} evidence "
                        f"quote is {len(quote)} characters (> {MAX_QUOTE_CHARS}); "
                        "long quotes are more likely to cross a chunk boundary and "
                        "never match"
                    )
    return warnings


def _with_defaults(item) -> dict:
    if not isinstance(item, dict):
        return {"_invalid": item}
    nuggets = item.get("nuggets") or []
    return {
        **item,
        "expected_behavior": item.get("expected_behavior", "answer"),
        "topics": item.get("topics") or [],
        "nuggets": [
            {**n, "evidence": n.get("evidence") or []} if isinstance(n, dict) else n
            for n in nuggets
        ],
    }


def _problems(items: list[dict]) -> list[str]:
    problems = []
    seen_ids = set()
    for index, item in enumerate(items):
        if "_invalid" in item:
            problems.append(f"item {index + 1}: not an object")
            continue
        item_id = item.get("id") or f"item {index + 1}"
        if not item.get("id"):
            problems.append(f"{item_id}: missing id")
        elif item_id in seen_ids:
            problems.append(f"{item_id}: duplicate id")
        seen_ids.add(item_id)
        problems += [f"{item_id}: {p}" for p in _item_problems(item)]
    return problems


def _item_problems(item: dict) -> list[str]:
    problems = []
    if not (isinstance(item.get("question"), str) and item["question"].strip()):
        problems.append("missing question")
    if "key_facts" in item:
        problems.append("key_facts is the v1 format; use nuggets")
    for topic in item["topics"]:
        if topic not in TOPICS:
            problems.append(f"unknown topic {topic!r} (allowed: {', '.join(TOPICS)})")

    behaviour = item["expected_behavior"]
    if behaviour not in BEHAVIOURS:
        problems.append(f"expected_behavior {behaviour!r} (allowed: answer, refuse)")
    nuggets = item["nuggets"]
    if behaviour == "refuse" and nuggets:
        problems.append("refuse items must not have nuggets")
    if behaviour == "answer":
        if not any(
            isinstance(n, dict) and n.get("importance") == "vital" for n in nuggets
        ):
            problems.append("needs at least one vital nugget")
    for n, nugget in enumerate(nuggets, start=1):
        problems += _nugget_problems(nugget, n)
    return problems


def _nugget_problems(nugget, n: int) -> list[str]:
    if not isinstance(nugget, dict):
        return [f"nugget {n} is not an object"]
    problems = []
    if not (isinstance(nugget.get("text"), str) and nugget["text"].strip()):
        problems.append(f"nugget {n} has no text")
    importance = nugget.get("importance")
    if importance not in IMPORTANCE:
        problems.append(f"nugget {n} has importance {importance!r} (vital or okay)")
    evidence = nugget["evidence"]
    if not isinstance(evidence, list):
        problems.append(f"nugget {n} evidence must be a list of quotes")
        return problems
    if importance == "vital" and not evidence:
        problems.append(f"vital nugget {n} has no evidence quote")
    if any(not (isinstance(q, str) and q.strip()) for q in evidence):
        problems.append(f"nugget {n} has an empty evidence quote")
    return problems
