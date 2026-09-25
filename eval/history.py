"""Eval run history: one JSON line per run in eval/history/runs.jsonl (committed).

Each record holds what was tested and the scores, so runs can be compared over
time. Full reports with generated answers stay in eval/reports/ (gitignored).

CLI:
    uv run python -m eval.history import-reports [--since YYYYMMDD]
        Backfill history from existing report_*.json files.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = _PROJECT_ROOT / "eval" / "history" / "runs.jsonl"
_CHAINS_DIR = _PROJECT_ROOT / "graph" / "chains"
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
                "expected_behavior": r.get("expected_behavior", "answer"),
                "pass": r.get("pass"),
                "error_type": r.get("error_type"),
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


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo: Path = _PROJECT_ROOT) -> dict:
    """Commit and branch the run used, and whether tracked files had uncommitted
    changes (then the commit alone does not say what ran). Nulls outside git."""
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git(repo, "rev-parse", "--short", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def prompt_fingerprints(chains_dir: Path = _CHAINS_DIR) -> dict[str, str]:
    """Hash of each chain module, where the prompts live. Tells which prompt
    changed between two runs even when one ran on uncommitted code."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        for path in sorted(chains_dir.glob("*.py"))
    }


def import_reports(
    reports_dir: Path,
    path: Path = DEFAULT_HISTORY_PATH,
    since: str | None = None,
) -> int:
    """Append report_*.json runs not yet in the history; returns how many.

    since: skip runs whose id sorts before it (e.g. "20260916").
    Reports written before runs carried a header are labelled "imported", with
    git and config unknown and no content_hash (grading targets were not saved).
    """
    known = {r.get("run_id") for r in load_runs(path)}
    imported = 0
    for report in sorted(reports_dir.glob("report_*.json")):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        run_id = data.get("run_id")
        if not run_id or run_id in known or (since and run_id < since):
            continue
        results = data.get("results", [])
        dataset = data.get("dataset")
        if dataset is None:
            dataset = {**dataset_fingerprint(results), "content_hash": None}
        append_run(
            build_run_record(
                run_id=run_id,
                summary=data.get("summary", {}),
                results=results,
                dataset=dataset,
                config=data.get("config") or {},
                git=data.get("git"),
                label=data.get("label") or "imported",
                notes=data.get("notes"),
                duration_sec=data.get("duration_sec"),
                report_file=report.name,
            ),
            path,
        )
        known.add(run_id)
        imported += 1
    return imported


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the eval run history.")
    sub = parser.add_subparsers(dest="command", required=True)
    backfill = sub.add_parser(
        "import-reports", help="Backfill the history from existing report JSON files"
    )
    backfill.add_argument(
        "--reports-dir", type=Path, default=_PROJECT_ROOT / "eval" / "reports"
    )
    backfill.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_PATH)
    backfill.add_argument(
        "--since", default=None, help="Only runs from this date on (YYYYMMDD)"
    )
    args = parser.parse_args()

    n = import_reports(args.reports_dir, args.history_file, since=args.since)
    print(f"Imported {n} run(s) into {args.history_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
