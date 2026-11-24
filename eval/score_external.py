"""Judge and score answers from an external system with this project's judges.

Built for the joint evaluation with GEUS (Scientific Friday Bar, 27 Nov 2026): an
OpenCode harness answers the held-out questions from the Markdown export
(scripts/export_markdown.py), and its answers are scored exactly like a run of
the RAG pipeline: same golden nuggets, same judge prompts and votes, same pass rule.

Input: one JSON object per line and question, as asked for in export/README.md:

    {"id": "...", "answer": "...",
     "quotes": [{"file": "dk/....md", "quote": "..."}],
     "trace": [...], "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}

The "context" the answer is judged against (groundedness) and that the evidence
metrics look in is what the agent read, rebuilt from its trace. A trace step may be
- a string, or an object with "output" / "content" / "result": that text as read;
- an object with "file" (or "path") and a line range: "start"/"end",
  "offset"/"limit" (OpenCode's read tool), or "lines": "120-180"; the lines are
  read from the export. A file without a range counts as read in full.
Without a usable trace the quotes stand in as context, and the run notes that
the search/answer split is then not meaningful.

Tables: the golden evidence quotes for table rows were copied from the index,
where docling linearises a row as "row, column = value. row, column = value".
The export has Markdown tables, so for the evidence metrics (not for the judge)
every Markdown table in the context is also written in that form.

Usage:
    # check the file first: no judge, no cost
    uv run python -m eval.score_external ANSWERS.jsonl --check
    # judge and record (same env as other runs: EVAL_GRADER_PROVIDER, EVAL_JUDGE_MODEL)
    uv run python -m eval.score_external ANSWERS.jsonl --label geus-harness-qwen \\
        --model Qwen3.6-35B-A3B-FP8 --system "GEUS OpenCode harness"
    # second judge on the same answers, as for any run:
    uv run python -m eval.run_eval --golden eval/data/golden_test.json \\
        --split confirm --rescore RUN_ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from eval.golden import load_golden
from eval.history import DEFAULT_HISTORY_PATH, dataset_fingerprint, git_info
from eval.scoring import METRICS_VERSION, evidence_found, normalize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "eval" / "data" / "golden_test.json"
DEFAULT_CORPUS = ROOT / "export" / "markdown"
EXTERNAL_PROVIDER = "external"

# --- context from the trace -----------------------------------------------------------

_RANGE = re.compile(r"^\s*(\d+)\s*[-–:]\s*(\d+)\s*$")


class CorpusFiles:
    """Lines of the exported Markdown files, found by relative path or file name."""

    def __init__(self, root: Path):
        self.root = root
        self._by_name = {p.name: p for p in root.rglob("*.md")} if root.exists() else {}
        self._cache: dict[Path, list[str]] = {}

    def _resolve(self, name: str) -> Path | None:
        name = name.strip().replace("\\", "/")
        candidates = [self.root / name, self.root / name.split("markdown/")[-1]]
        path = next((p for p in candidates if p.is_file()), None)
        return path or self._by_name.get(Path(name).name)

    def relative(self, name: str) -> str:
        path = self._resolve(name)
        return path.relative_to(self.root).as_posix() if path else name

    def lines(self, name: str) -> list[str] | None:
        path = self._resolve(name)
        if path is None:
            return None
        if path not in self._cache:
            self._cache[path] = path.read_text(encoding="utf-8").splitlines()
        return self._cache[path]


    def files_under(self, path: str | None, include: str | None) -> list[tuple[str, Path]]:
        """(relative name, path) of the export's files a grep step searched.

        path is relative to the corpus (".", "dk", a file) or an absolute path on the
        other machine ending in the corpus folder; include is a ripgrep glob, matched
        against the file name when it has no "/", else against the relative path.
        """
        import fnmatch

        base = (path or ".").replace("\\", "/")
        if base.startswith("/"):
            base = "." if base.rstrip("/").endswith("corpus") else base.split("corpus/")[-1]
        target = self.root / base
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = sorted(target.rglob("*.md"))
        else:
            candidates = [self._by_name[Path(base).name]] if Path(base).name in self._by_name else []
        found = []
        for p in candidates:
            rel = p.relative_to(self.root).as_posix()
            if include:
                pattern = include.replace("\\", "/")
                # ripgrep matches a glob with "/" against the path below the
                # searched folder: include "dk/*.md" under path "dk" finds nothing
                below = p.relative_to(target).as_posix() if target.is_dir() else p.name
                hit = (
                    fnmatch.fnmatch(below, pattern)
                    if "/" in pattern
                    else fnmatch.fnmatch(p.name, pattern)
                )
                if not hit:
                    continue
            found.append((rel, p))
        return found


# OpenCode's grep tool returns at most this many matches
GREP_MAX_MATCHES = 100


def replay_grep(step: dict, corpus: CorpusFiles) -> tuple[list[tuple[str, int]], int] | None:
    """The lines a grep step showed, re-run on the export: ([(file, line)], matches).

    The trace records the search (pattern, path, include) but not what it returned;
    the agent saw the matching lines, so they belong to what it read. Python's re
    stands in for ripgrep; None if the pattern does not compile. OpenCode shows at
    most GREP_MAX_MATCHES matches (it orders them by file modification time, which
    cannot be replayed; here they come in file order).
    """
    try:
        regex = re.compile(step["pattern"])
    except (re.error, KeyError, TypeError):
        return None
    hits: list[tuple[str, int]] = []
    total = 0
    for rel, _path in corpus.files_under(step.get("path"), step.get("include")):
        for n, line in enumerate(corpus.lines(rel) or [], start=1):
            if regex.search(line):
                total += 1
                if total <= GREP_MAX_MATCHES:
                    hits.append((rel, n))
    return hits, total


def _line_range(step: dict, n_lines: int) -> tuple[int, int]:
    """1-based inclusive range of a read step; the whole file if none is given."""
    lines = step.get("lines")
    if isinstance(lines, list) and len(lines) == 2:
        return int(lines[0]), int(lines[1])
    if isinstance(lines, str) and (m := _RANGE.match(lines)):
        return int(m.group(1)), int(m.group(2))
    if step.get("start") is not None:
        start = int(step["start"])
        end = int(step.get("end") or start)
        return start, end
    if step.get("offset") is not None:
        start = int(step["offset"]) + 1  # OpenCode's read: 0-based offset
        return start, start + int(step.get("limit") or 2000) - 1
    return 1, n_lines


def _render_seen(seen: dict[str, set[int]], corpus: CorpusFiles) -> list[str]:
    """Every line the agent saw, once: per file in order of first access, as runs of
    consecutive lines. Runs of spaces (docling pads table cells) are squeezed."""
    blocks = []
    for rel, numbers in seen.items():
        lines = corpus.lines(rel) or []
        # The file's title (line 1, e.g. "# ... (BEK nr 1385 af 18.11.2025)") heads
        # every index chunk, so the judge can check a cited order number for the
        # pipeline; it does the same here, whatever offset the agent read from.
        if lines and lines[0].startswith("# "):
            numbers = numbers | {1}
        run: list[int] = []
        for n in sorted(numbers) + [None]:
            if n is not None and (not run or n == run[-1] + 1):
                run.append(n)
                continue
            if run:
                text = "\n".join(lines[i - 1] for i in run if 0 < i <= len(lines))
                text = re.sub(r" {2,}", " ", text)
                blocks.append(f"[{rel}, lines {run[0]}-{run[-1]}]\n{text}")
            run = [n] if n is not None else []
    return blocks


# Characters of context the judge is given. The judge models take about 128k
# tokens; the RAG pipeline's contexts are 12k-31k characters.
MAX_JUDGE_CONTEXT = 150_000


def judge_context(record: dict, corpus: CorpusFiles, full: str) -> tuple[str, bool]:
    """The context for the judge: everything seen if it fits, else the passages the
    agent read (grep hit lines left out), cut at MAX_JUDGE_CONTEXT."""
    if len(full) <= MAX_JUDGE_CONTEXT:
        return full, False
    reads = {"trace": [s for s in record.get("trace") or [] if not (isinstance(s, dict) and s.get("tool") == "grep")]}
    text, _, _ = rebuild_context(reads, corpus)
    note = "\n\n[grep hit lines left out: the agent's full context exceeded the judge's limit]"
    return text[: MAX_JUDGE_CONTEXT - len(note)] + note, True


def rebuild_context(record: dict, corpus: CorpusFiles) -> tuple[str, str, list[str]]:
    """(context, source, problems): what the agent saw.

    Lines from read steps and from replayed grep output are collected per file and
    shown once each: an agent that searches the same document twenty times has
    still seen each passage, not twenty copies of it. Text given in the trace
    itself (an "output") is kept as it is. source is "trace" or "quotes" (no usable
    trace: the quotes stand in).
    """
    pieces: list[str] = []
    seen: dict[str, set[int]] = {}
    problems: list[str] = []
    for step in record.get("trace") or []:
        if isinstance(step, str):
            pieces.append(step)
            continue
        if not isinstance(step, dict):
            continue
        text = next(
            (step[k] for k in ("output", "content", "result") if step.get(k)), None
        )
        if text:
            pieces.append(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False))
            continue
        if step.get("tool") == "grep":
            if step.get("error") or not step.get("pattern"):
                continue
            replayed = replay_grep(step, corpus)
            if replayed is None:
                problems.append(f"grep pattern not replayable: {step.get('pattern')}")
                continue
            hits, total = replayed
            recorded = step.get("n_matches")
            # the recorded count runs about a third above a line count (it seems to
            # count more than matching lines); flag only a search that found
            # nothing on one side
            if recorded is not None and (recorded == 0) != (total == 0):
                problems.append(
                    f"grep replay found {total} matches, the trace says {recorded}: "
                    f"{step['pattern'][:60]}"
                )
            for rel, n in hits:
                seen.setdefault(rel, set()).add(n)
            continue
        if step.get("tool") == "glob":
            continue  # file names only
        name = step.get("file") or step.get("path")
        if not name:
            continue
        lines = corpus.lines(str(name))
        if lines is None:
            problems.append(f"trace file not in the export: {name}")
            continue
        rel = corpus.relative(str(name))
        start, end = _line_range(step, len(lines))
        seen.setdefault(rel, set()).update(range(max(start, 1), min(end, len(lines)) + 1))
    pieces += _render_seen(seen, corpus)
    if pieces:
        return "\n\n".join(pieces), "trace", problems
    quotes = [q.get("quote", "") if isinstance(q, dict) else str(q) for q in record.get("quotes") or []]
    return "\n\n".join(q for q in quotes if q), "quotes", problems


# --- tables in the index's linearised form ------------------------------------------------


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def linearise_tables(text: str) -> str:
    """Markdown tables of text as docling's chunk text: 'row, column = value. ...'.

    One line per table row: the first cell names the row, the header row names the
    columns. Used only to find golden evidence quotes, which were copied from the
    index's chunks.
    """
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        is_table = lines[i].lstrip().startswith("|") and re.match(
            r"^\s*\|?\s*:?-{3,}", lines[i + 1]
        )
        if not is_table:
            i += 1
            continue
        header = _cells(lines[i])
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            row = _cells(lines[i])
            parts = [
                f"{row[0]}, {header[j]} = {row[j]}"
                for j in range(1, min(len(row), len(header)))
                if row[j]
            ]
            if parts:
                out.append(". ".join(parts))
            i += 1
    return "\n".join(out)


def flatten_markdown(text: str) -> str:
    """The text without Markdown syntax, as one run of words.

    The Danish orders were indexed from XML as plain text: table cells and rows
    joined by spaces ("Co-58 1 100 Co-60 0,1 10"), list items run on ("... arrangement.
    5) Transport ..."). The export writes the same text as tables, bullets and
    headings; this undoes that for evidence matching.
    """
    kept = []
    for line in text.splitlines():
        if re.match(r"^\s*\|?\s*:?-{3,}", line):  # table separator row
            continue
        if line.lstrip().startswith("|"):
            # the first export sent to GEUS (23 Nov, 1,903,438 bytes) joined merged
            # header cells with ": " ("Primært referenceniveau: P_KA"); the index
            # joins them with a space
            line = line.replace(": ", " ")
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        # "cm^2" in the export, "cm²" (normalised "cm2") in older evidence quotes
        line = line.replace("**", "").replace("|", " ").replace("^", "")
        kept.append(line)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def evidence_context(context: str) -> str:
    """The context plus two re-renderings, for evidence matching only (never judged):
    tables linearised as in the IAEA chunks, and Markdown flattened as in the
    Danish XML chunks."""
    extra = [t for t in (linearise_tables(context), flatten_markdown(context)) if t]
    return "\n\n".join([context, *extra])


# --- loading ---------------------------------------------------------------------------


def load_answers(path: Path) -> dict[str, dict]:
    answers: dict[str, dict] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name} line {n}: not JSON ({e})") from e
        if not record.get("id"):
            raise ValueError(f"{path.name} line {n}: no id")
        if record["id"] in answers:
            raise ValueError(f"{path.name} line {n}: {record['id']} appears twice")
        answers[record["id"]] = record
    return answers


def build_outputs(
    golden: list[dict], answers: dict[str, dict], corpus: CorpusFiles
) -> tuple[dict[str, dict], list[str]]:
    """Graph-output records for score_item / save_outputs, plus problems found."""
    outputs: dict[str, dict] = {}
    problems: list[str] = []
    for item in golden:
        record = answers.get(item["id"])
        if record is None:
            problems.append(f"{item['id']}: no answer (scored as failed)")
            outputs[item["id"]] = {"generation": "", "error": "no answer from the external system"}
            continue
        asked = record.get("question")
        if asked and normalize(asked) != normalize(item["question"]):
            problems.append(f"{item['id']}: question text differs from the golden set")
        context, source, context_problems = rebuild_context(record, corpus)
        problems += [f"{item['id']}: {p}" for p in context_problems]
        judged, trimmed = judge_context(record, corpus, context)
        outputs[item["id"]] = {
            "generation": record.get("answer") or "",
            # what the judge checks the answer against
            "context_used_for_generation": judged,
            # everything the agent saw, for the evidence metrics
            "context_seen": context,
            "context_trimmed_for_judge": trimmed,
            "context_source": source,
            "initial_documents": [],
            "sufficient": None,
            "retrieval_warning": None,
            "web_search_attempted": False,
            "node_path": ["external"],
            "error": record.get("error"),
            "external": {
                k: record.get(k)
                for k in ("quotes", "tokens_in", "tokens_out", "seconds", "model", "harness")
                if record.get(k) is not None
            },
        }
    extra = sorted(set(answers) - {i["id"] for i in golden})
    if extra:
        problems.append(f"answers for ids not in the golden selection: {', '.join(extra)}")
    return outputs, problems


# --- scoring ---------------------------------------------------------------------------


def _scores(item: dict, run: dict, verdict: dict | None) -> dict:
    from eval.scoring import score_item

    output = {
        "initial_documents": [],
        "context": evidence_context(
            run.get("context_seen") or run.get("context_used_for_generation") or ""
        ),
        "sufficient": None,
        "retrieval_warning": None,
    }
    scores = score_item(item, output, verdict)
    # there is no first retrieval in a harness: the metric would read 0, not "n/a"
    scores["evidence_recall_initial"] = None
    scores["grade_documents_correct"] = None
    return scores


def check(golden: list[dict], outputs: dict[str, dict]) -> None:
    """Per question without a judge: context size and source, vital evidence read."""
    sources: dict[str, int] = {}
    for item in golden:
        run = outputs[item["id"]]
        sources[run.get("context_source", "none")] = sources.get(run.get("context_source", "none"), 0) + 1
        context = evidence_context(run.get("context_seen") or "")
        vital = [n for n in item.get("nuggets") or [] if n["importance"] == "vital"]
        found = sum(evidence_found(n["evidence"], [context]) for n in vital)
        evidence = f"vital evidence read {found}/{len(vital)}" if vital else "refusal item"
        trimmed = " TRIMMED" if run.get("context_trimmed_for_judge") else ""
        print(
            f"{item['id'][:55]:55} {len(run.get('generation') or ''):>6} ch answer "
            f"{len(run.get('context_seen') or ''):>8} ch seen "
            f"{len(run.get('context_used_for_generation') or ''):>7} ch judged{trimmed}; {evidence}"
        )
    print(f"context sources: {sources}")


def main() -> int:
    from dotenv import load_dotenv

    # Judge settings and API keys come from .env, as for eval.run_eval.
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("answers", type=Path, help="JSONL with one answer per question")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--split", default="confirm", help="tune, confirm or all")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--check", action="store_true", help="no judge: check the file and the context")
    p.add_argument("--model", default="unknown", help="answer model of the external system")
    p.add_argument("--system", default="external system", help="name of the external system")
    p.add_argument("--label", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "reports")
    p.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY_PATH)
    p.add_argument("--no-history", action="store_true")
    args = p.parse_args()

    golden = load_golden(args.golden)
    if args.split != "all":
        from eval.split import select_split

        golden = select_split(golden, args.split)
    answers = load_answers(args.answers)
    outputs, problems = build_outputs(golden, answers, CorpusFiles(args.corpus))
    for problem in problems:
        print(f"Warning: {problem}", file=sys.stderr)
    if args.check:
        check(golden, outputs)
        return 0

    from langchain_core.callbacks import get_usage_metadata_callback

    from eval.dashboard import write_dashboard
    from eval.judge import judge_item
    from eval.run_eval import (
        _invoke_with_retry,
        _judge_llm,
        _render_report,
        _result,
        _usage,
        _warn_if_self_judged,
        summarize,
    )
    from eval.saved_runs import SavedRuns
    from graph.llm_factory import get_llm

    started = time.monotonic()
    judge_llm, judge_info = _judge_llm(get_llm(), EXTERNAL_PROVIDER, args.model)
    if not (judge_info["judge_provider"] and judge_info["judge_provider"] != EXTERNAL_PROVIDER):
        print("Set EVAL_GRADER_PROVIDER and EVAL_JUDGE_MODEL for the judge.", file=sys.stderr)
        return 1
    _warn_if_self_judged(judge_info)

    runs = SavedRuns(args.output_dir, args.history_file)
    run_id = runs.claim_run_id()
    runs.save(run_id, outputs)
    results = []
    with get_usage_metadata_callback() as judging:
        for n, item in enumerate(golden, start=1):
            run = outputs[item["id"]]
            if run.get("error"):
                verdict = None
                scores = {**_scores(item, run, None), "error_type": "graph_error", "pass": False}
            else:
                verdict = _invoke_with_retry(
                    judge_item,
                    item,
                    run["generation"],
                    run.get("context_used_for_generation") or "",
                    judge_llm,
                    votes=judge_info["judge_votes"],
                )
                scores = _scores(item, run, verdict)
            results.append(_result(item, run, scores, verdict))
            print(f"  [{n}/{len(golden)}] judged {item['id']}: {scores['error_type']}", file=sys.stderr)

    tokens = {
        k: sum(int(r.get(k) or 0) for r in answers.values()) for k in ("tokens_in", "tokens_out")
    }
    seconds = [float(r["seconds"]) for r in answers.values() if r.get("seconds") is not None]
    sources = [outputs[i["id"]].get("context_source") for i in golden]
    header = {
        "label": args.label,
        "notes": args.notes,
        "git": git_info(),
        "dataset": dataset_fingerprint(golden),
        "config": {
            "retrieval_only": False,
            "llm_provider": EXTERNAL_PROVIDER,
            "llm_model": args.model,
            "external_system": args.system,
            "answers_file": args.answers.name,
            "answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest()[:12],
            "context_from_trace": sources.count("trace"),
            "context_from_quotes": sources.count("quotes"),
            "context_trimmed_for_judge": sum(
                bool(outputs[i["id"]].get("context_trimmed_for_judge")) for i in golden
            ),
            "max_judge_context_chars": MAX_JUDGE_CONTEXT,
            "external_tokens": tokens,
            "external_median_seconds": sorted(seconds)[len(seconds) // 2] if seconds else None,
            "evidence_tables_linearised": True,
            **judge_info,
            "web_search": False,
            "metrics_version": METRICS_VERSION,
        },
        "usage": _usage(judge=judging),
        "duration_sec": round(time.monotonic() - started, 1),
    }
    summary = summarize(results)
    report, markdown = _render_report(summary, results, run_id, header)
    artifacts = runs.finish(report, markdown, record_history=not args.no_history)
    if not args.no_history:
        write_dashboard(args.output_dir / "dashboard.html", args.history_file, reports_dir=args.output_dir)
        print(f"Run recorded: {run_id}")
    print(f"Report written: {artifacts.markdown_report}")
    if summary["pass_rate"] is not None:
        lo, hi = summary.get("pass_rate_interval") or (None, None)
        passed = sum(1 for r in results if r["pass"])
        print(f"Pass: {passed}/{summary['judged']}" + (f" (95 % {lo:.0%}–{hi:.0%})" if lo is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
