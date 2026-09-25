"""Local HTML dashboard for comparing eval runs over time.

Reads eval/history/runs.jsonl and writes one self-contained HTML file.

    uv run python -m eval.dashboard [--baseline RUN_ID] [--output PATH] [--open]

All comparison logic lives here in Python (tested); the page's script only
draws what dashboard_data() computed.
"""

import argparse
import json
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path

from eval.history import DEFAULT_HISTORY_PATH, load_runs

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "eval" / "reports"
_TEMPLATE = Path(__file__).with_name("dashboard_template.html")
_DATA_PLACEHOLDER = "__DASHBOARD_DATA__"

# Topic/language/source groups with fewer questions than this say little on
# their own; the dashboard greys them out.
LOW_N = 3

_TAG_DIMENSIONS = ("topics", "language", "source")
_UNTAGGED = "untagged"


def dashboard_data(
    runs: list[dict],
    reports_dir: Path | None = None,
    baseline_run_id: str | None = None,
) -> dict:
    """Everything the dashboard shows, computed from the run history.

    runs: history records, any order.
    reports_dir: where local full reports live, for answer previews (optional).
    baseline_run_id: compare every run in its question set against this run;
    by default the latest run labelled "baseline", else the set's first run.
    """
    runs = sorted(runs, key=lambda r: r.get("run_id", ""))
    by_set: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_set[run["dataset"]["questions_hash"]].append(run)

    sets = [
        _question_set(qhash, set_runs, baseline_run_id)
        for qhash, set_runs in by_set.items()
    ]
    sets.sort(key=lambda s: s["run_ids"][-1], reverse=True)
    return {
        "metrics": _metric_names(runs),
        "runs": {r["run_id"]: _run_view(r, reports_dir) for r in runs},
        "sets": sets,
    }


def _metric_names(runs: list[dict]) -> list[str]:
    names: list[str] = []
    for run in runs:
        for result in run.get("results", []):
            for name in result.get("metrics", {}):
                if name not in names:
                    names.append(name)
    return names


def _run_view(run: dict, reports_dir: Path | None) -> dict:
    previews = _load_previews(run, reports_dir)
    return {
        **{k: v for k, v in run.items() if k != "results"},
        "results": {
            r["id"]: {
                "pass": r.get("pass"),
                "metrics": r.get("metrics", {}),
                "web_search_attempted": r.get("web_search_attempted"),
                "retrieval_warning": r.get("retrieval_warning"),
                "preview": previews.get(r["id"]),
            }
            for r in run.get("results", [])
        },
    }


def _load_previews(run: dict, reports_dir: Path | None) -> dict[str, str]:
    """Generated-answer previews from the local full report, if it still exists."""
    if reports_dir is None or not run.get("report_file"):
        return {}
    try:
        report = json.loads((reports_dir / run["report_file"]).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        r.get("id", ""): r.get("generation_preview") for r in report.get("results", [])
    }


def _question_set(
    questions_hash: str, runs: list[dict], baseline_run_id: str | None
) -> dict:
    latest = runs[-1]
    baseline = _pick_baseline(runs, baseline_run_id)
    comparisons = {}
    markers = []
    for i, run in enumerate(runs):
        previous = runs[i - 1] if i > 0 else None
        against_baseline = (
            baseline if baseline and baseline["run_id"] < run["run_id"] else None
        )
        comparisons[run["run_id"]] = {
            "previous": compare_runs(previous, run) if previous else None,
            "baseline": (
                compare_runs(against_baseline, run) if against_baseline else None
            ),
        }
        if previous:
            reasons = _incomparability(previous, run)
            if reasons:
                markers.append({"run_id": run["run_id"], "reasons": reasons})
    return {
        "questions_hash": questions_hash,
        "n_items": latest["dataset"].get("n_items"),
        "questions": [
            {
                "id": r["id"],
                "question": r.get("question", ""),
                "topics": r.get("topics") or [],
                "language": r.get("language"),
                "source": r.get("source"),
            }
            for r in latest.get("results", [])
        ],
        "run_ids": [r["run_id"] for r in runs],
        "baseline_run_id": baseline["run_id"] if baseline else None,
        "markers": markers,
        "comparisons": comparisons,
        "breakdown": {r["run_id"]: tag_breakdown(r) for r in runs},
    }


def _pick_baseline(runs: list[dict], baseline_run_id: str | None) -> dict | None:
    if baseline_run_id:
        chosen = [r for r in runs if r["run_id"] == baseline_run_id]
        if chosen:
            return chosen[0]
    labelled = [r for r in runs if r.get("label") == "baseline"]
    return labelled[-1] if labelled else runs[0]


def _incomparability(before: dict, after: dict) -> list[str]:
    """Why scores on either side of this pair do not mean the same thing."""
    reasons = []
    hashes = (
        before["dataset"].get("content_hash"),
        after["dataset"].get("content_hash"),
    )
    if None not in hashes and hashes[0] != hashes[1]:
        reasons.append("grading targets changed")
    versions = (
        (before.get("config") or {}).get("metrics_version"),
        (after.get("config") or {}).get("metrics_version"),
    )
    if None not in versions and versions[0] != versions[1]:
        reasons.append("scoring changed")
    return reasons


def compare_runs(base: dict, run: dict) -> dict:
    """How `run` differs from `base`: flipped questions, score and setting changes."""
    base_results = {r["id"]: r for r in base.get("results", [])}
    regressions, improvements, changed = [], [], []
    for result in run.get("results", []):
        before = base_results.get(result["id"])
        if before is None:
            continue
        entry = {
            "id": result["id"],
            "question": result.get("question", ""),
            "before": before.get("metrics", {}),
            "after": result.get("metrics", {}),
        }
        if before.get("pass") and not result.get("pass"):
            regressions.append(entry)
        elif not before.get("pass") and result.get("pass"):
            improvements.append(entry)
        elif entry["before"] != entry["after"]:
            changed.append(entry)

    warnings = _incomparability(base, run)
    base_config, run_config = base.get("config") or {}, run.get("config") or {}
    if base_config and run_config:
        config_diff = _diff(_flatten(base_config), _flatten(run_config))
    else:
        config_diff = None
        for r in (base, run):
            if not r.get("config"):
                warnings.append(f"settings not recorded for run {r['run_id']}")
    for r in (base, run):
        if (r.get("git") or {}).get("dirty"):
            warnings.append(f"run {r['run_id']} had uncommitted changes")

    return {
        "base_run_id": base["run_id"],
        "run_id": run["run_id"],
        "verdict": (
            f"{_plural(len(regressions), 'regression')}, "
            f"{_plural(len(improvements), 'improvement')}"
        ),
        "regressions": regressions,
        "improvements": improvements,
        "changed": changed,
        "summary_delta": {
            key: run["summary"][key] - base["summary"][key]
            for key in run.get("summary", {})
            if _is_number(run["summary"][key])
            and _is_number(base.get("summary", {}).get(key))
        },
        "config_diff": config_diff,
        "commits": [
            (base.get("git") or {}).get("commit"),
            (run.get("git") or {}).get("commit"),
        ],
        "warnings": warnings,
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _diff(before: dict, after: dict) -> list[dict]:
    return [
        {"field": key, "before": before.get(key), "after": after.get(key)}
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    ]


def tag_breakdown(run: dict) -> dict:
    """Pass rate and metric means per topic, language and source.

    A question with several topics counts in each. Groups smaller than LOW_N
    are flagged, since one question flipping swings them completely.
    """
    groups: dict[str, dict[str, list[dict]]] = {
        d: defaultdict(list) for d in _TAG_DIMENSIONS
    }
    for result in run.get("results", []):
        for dimension in _TAG_DIMENSIONS:
            value = result.get(dimension)
            values = value if isinstance(value, list) else [value]
            for tag in [v for v in values if v] or [_UNTAGGED]:
                groups[dimension][tag].append(result)
    return {
        dimension: {tag: _group_stats(results) for tag, results in sorted(tags.items())}
        for dimension, tags in groups.items()
    }


def _group_stats(results: list[dict]) -> dict:
    n = len(results)
    names = list(dict.fromkeys(m for r in results for m in r.get("metrics", {})))
    return {
        "n": n,
        "pass_rate": sum(1 for r in results if r.get("pass")) / n,
        "means": {m: sum(r["metrics"].get(m, 0.0) for r in results) / n for m in names},
        "low_n": n < LOW_N,
    }


def render_html(data: dict) -> str:
    """The dashboard page with `data` embedded as JSON.

    "<" is escaped so no text in the history (a question, a label) can close
    the data's <script> element and inject markup.
    """
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return _TEMPLATE.read_text(encoding="utf-8").replace(_DATA_PLACEHOLDER, payload)


def write_dashboard(
    output: Path,
    history_file: Path = DEFAULT_HISTORY_PATH,
    reports_dir: Path | None = _REPORTS_DIR,
    baseline_run_id: str | None = None,
) -> Path:
    data = dashboard_data(
        load_runs(history_file),
        reports_dir=reports_dir,
        baseline_run_id=baseline_run_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a local HTML dashboard comparing eval runs."
    )
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_REPORTS_DIR,
        help="Local full reports, for answer previews",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="RUN_ID",
        help="Compare runs against this run (default: latest run labelled 'baseline')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPORTS_DIR / "dashboard.html",
        help="Where to write the page (default: eval/reports/dashboard.html)",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the page in the browser"
    )
    args = parser.parse_args()

    path = write_dashboard(
        args.output, args.history_file, args.reports_dir, args.baseline
    )
    print(f"Dashboard written: {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
