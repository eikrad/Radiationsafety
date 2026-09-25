"""Eval run history: one JSON line per run in eval/history/runs.jsonl (committed).

Each record holds what was tested and the scores, so runs can be compared over
time. Full reports with generated answers stay in eval/reports/ (gitignored).
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = _PROJECT_ROOT / "eval" / "history" / "runs.jsonl"
SCHEMA_VERSION = 1


def _sha256_short(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def dataset_fingerprint(items: list[dict]) -> dict:
    """Fingerprint the golden items actually run.

    questions_hash: ids + questions only; runs with equal value are comparable.
    content_hash: also covers expected answers and key facts, so it changes when
    grading targets change even though the questions stay the same.
    Tags (topics, language, source) are left out: retagging keeps runs comparable.
    """
    questions = [[i.get("id", ""), i["question"]] for i in items]
    content = [
        [i.get("id", ""), i["question"], i.get("expected_answer"), i.get("key_facts")]
        for i in items
    ]
    return {
        "questions_hash": _sha256_short(questions),
        "content_hash": _sha256_short(content),
        "n_items": len(items),
    }


def build_run_record(
    *,
    run_id: str,
    summary: dict,
    results: list[dict],
    dataset: dict,
    config: dict,
    git: dict | None,
    label: str | None = None,
    notes: str | None = None,
    duration_sec: float | None = None,
    report_file: str | None = None,
) -> dict:
    """Assemble one history record. Generated answers are left out to keep it small."""
    return {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": _run_id_to_iso(run_id),
        "label": label,
        "notes": notes,
        "git": git,
        "dataset": dataset,
        "config": config,
        "duration_sec": round(duration_sec, 1) if duration_sec is not None else None,
        "report_file": report_file,
        "summary": summary,
        "results": [
            {
                "id": r.get("id", ""),
                "question": r.get("question", ""),
                "topics": r.get("topics") or [],
                "language": r.get("language"),
                "source": r.get("source"),
                "pass": r.get("pass"),
                "metrics": r.get("metrics", {}),
                "web_search_attempted": r.get("web_search_attempted"),
                "retrieval_warning": bool(r.get("retrieval_warning")),
            }
            for r in results
        ],
    }


def _run_id_to_iso(run_id: str) -> str | None:
    """Run ids are UTC timestamps (YYYYmmdd_HHMMSS), as written by run_eval."""
    try:
        dt = datetime.strptime(run_id, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    return dt.isoformat()


def append_run(record: dict, path: Path = DEFAULT_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_runs(path: Path = DEFAULT_HISTORY_PATH) -> list[dict]:
    """All recorded runs, oldest first. Malformed lines are skipped."""
    if not path.exists():
        return []
    runs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sorted(runs, key=lambda r: r.get("run_id", ""))
