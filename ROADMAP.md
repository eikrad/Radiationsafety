# Evaluation Roadmap — Braintrust Feature Parity

This roadmap describes how to build the evaluation capabilities that Braintrust
offers as a managed platform, directly into this codebase. Each phase is
self-contained and adds a concrete, usable capability. Phases are ordered from
quickest win to most complex.

## Related Plans

- Playwright UI-E2E-Abdeckung (essentielle Interface-Flows, gemocktes API):
  `.cursor/plans/playwright_essential_e2e_975e1548.plan.md`

---

## Phase 1 — Continuous Scoring

**Goal:** Replace binary (0 or 1) metric scores with genuine 0.0–1.0 continuous
scores. Binary scoring throws away signal: a partially-correct answer scores the
same as a completely wrong one.

**Why it matters:** Regressions and improvements that don't cross the 0.5
threshold are invisible today. Continuous scores let you track slow drift and
measure the real impact of prompt or retrieval changes.

> **Note (v0.3.0):** `GradeGeneration` was refactored as part of the Reflexion
> implementation. The schema now has `passed: bool` + `missing_info: str`
> (replacing the old `grounded` + `answers_question` pair). Task 1 below should
> add `score: float` to the simplified schema, not the old one.

### Tasks

1. **Update grader prompts** (`graph/chains/generation_grader.py`,
   `graph/chains/context_sufficiency_grader.py`):
   - Add a `score: float` field (0.0–1.0) alongside `passed` in `GradeGeneration`
     and alongside `binary_score` in `GradeSufficiency`.
   - Update the prompt to instruct the LLM to assign a numeric confidence
     level, not just yes/no.
   - Example rubric for faithfulness: 1.0 = fully supported, 0.75 = mostly
     supported with minor gaps, 0.5 = borderline, 0.25 = mostly unsupported,
     0.0 = contradicted by context.
   - Keep `passed: bool` for the Reflexion routing logic (it reads `passed`,
     not the float score).

2. **Update `eval/metrics.py`** to read the numeric `score` field instead of
   casting `passed` to 1.0/0.0. Keep the boolean fallback for backwards
   compatibility when `score` is absent.

3. **Update `eval/run_eval.py`**:
   - Add standard deviation and median alongside mean in the summary.
   - Add a score histogram (bucketed 0.0–0.2, 0.2–0.4, etc.) to the Markdown
     report so you can see the distribution at a glance.

4. **Update pass/fail logic**:
   - Default threshold stays at 0.5, but now a score of 0.3 is more meaningful
     than a score of 0.0.
   - Add an optional `--warn-threshold` (e.g. 0.7) that marks items yellow in
     the report even when they pass.

**Estimated effort:** 1–2 days  
**Files touched:** `graph/chains/generation_grader.py`,
`graph/chains/context_sufficiency_grader.py`, `eval/metrics.py`,
`eval/run_eval.py`

---

## Phase 2 — Native Observability (Cost & Latency Tracking)

**Goal:** Record token counts, estimated cost, and per-node latency for every
eval run — stored alongside the existing JSON/Markdown reports. No external
service needed.

**Why it matters:** Right now you have no visibility into how expensive a run
is, which retrieval or generation step is slow, or how cost/latency trends over
time. This phase closes that gap without adding LangSmith as a dependency.

### Tasks

1. **Add a timing decorator / context manager** (`graph/tracing.py`, new file):
   - Wrap each LangGraph node with a lightweight timer that records wall-clock
     duration in milliseconds.
   - Store results in a `run_metadata` dict that travels through `GraphState`.

2. **Add token counting** to each LLM call site:
   - LangChain's `ChatModel` responses include `response_metadata.usage` (input
     tokens, output tokens). Extract these after each call.
   - Accumulate totals in `run_metadata`.

3. **Add a cost estimator** (`eval/cost.py`, new file):
   - Maintain a small dict of `{provider: {model: (input_$/1k, output_$/1k)}}`.
   - Compute estimated cost from token counts. Mark as "estimated" in the
     report since pricing changes.

4. **Extend the eval report schema** (`eval/run_eval.py`):
   - Per-question: add `latency_ms`, `input_tokens`, `output_tokens`,
     `estimated_cost_usd`.
   - Summary: add total cost, mean latency, slowest question, highest-cost
     question.

5. **Print a cost/latency table** in the Markdown report after the per-question
   breakdown.

**Estimated effort:** 3–5 days  
**Files touched:** `graph/state.py`, `graph/graph.py`, `graph/nodes/*.py`,
`eval/run_eval.py`, new `graph/tracing.py`, new `eval/cost.py`

---

## Phase 3 — CI/CD Regression Gating

**Goal:** Run eval automatically on every pull request via GitHub Actions and
block merges when scores regress below the current baseline.

**Why it matters:** Right now, a PR that quietly breaks faithfulness from 0.85
to 0.60 will be merged with no warning. This phase makes metric regressions
visible on every PR before merge.

### Tasks

1. **Create a baseline file** (`eval/data/baseline.json`):
   - Schema: `{"faithfulness": float, "answer_relevance": float,
     "context_precision": float, "context_recall": float, "recorded_at":
     ISO-date}`.
   - Commit an initial baseline generated from the current golden set on
     `master`.
   - Add a CLI subcommand `eval.run_eval --update-baseline` that overwrites
     this file and prints a diff.

2. **Write a comparison script** (`eval/compare_baseline.py`):
   - Reads the latest report JSON and `baseline.json`.
   - For each metric, computes delta and flags as regression if delta < `-0.05`
     (configurable via `--tolerance`).
   - Exits with code 1 on regression, 0 on pass. Prints a human-readable diff
     table.

3. **Add a GitHub Actions workflow** (`.github/workflows/eval.yml`):
   ```
   Trigger: pull_request (paths: graph/**, eval/**, prompts/**)
   Steps:
     1. Checkout + uv sync
     2. Run ingestion (restore from cache if unchanged)
     3. uv run python -m eval.run_eval --limit 20 --no-web-search
     4. uv run python -m eval.compare_baseline
     5. Upload report artifact
   ```
   - Use `--limit 20` for PR checks (fast) and full 80-item set on merge to
     `master`.
   - Cache the `.chroma` vector store by hashing `documents/` so ingestion only
     runs when documents change.

4. **Post a PR comment** with the metric diff table using the GitHub Actions
   built-in token (`GITHUB_TOKEN`). No external service needed.

5. **Update baseline on merge**: add a step in `ci.yml` (the existing workflow)
   that runs `--update-baseline` after a successful merge to `master` and
   commits the updated `baseline.json`.

**Estimated effort:** 2–3 days  
**Files touched:** new `eval/data/baseline.json`, new `eval/compare_baseline.py`,
new `.github/workflows/eval.yml`, existing `.github/workflows/ci.yml`

---

## Phase 4 — Production Trace Capture & Dataset Growth

**Goal:** Log every real production query with its full context and response to
a local JSONL file. Provide a CLI command to promote any logged trace to the
golden dataset with one command.

**Why it matters:** The golden set today has 80 hand-crafted questions. Real
user queries surface edge cases and failure modes that synthetic data misses.
This phase lets the golden set grow from production usage.

### Tasks

1. **Add a trace logger** (`api/trace_log.py`, new file):
   - Append one JSON line per query to `eval/data/traces.jsonl` (gitignored by
     default).
   - Each line: `{id, timestamp, question, generation, context_used,
     retrieval_warning, web_search_attempted, latency_ms}`.
   - Wire it into `api/main.py` after the graph returns a result.
   - Make logging opt-in via `TRACE_LOG_ENABLED=true` in `.env` (off by
     default for privacy).

2. **Add a thumbs-down endpoint** (`api/main.py`):
   - `POST /api/feedback` with body `{trace_id: str, rating: "bad" | "good",
     comment: str | null}`.
   - Appends a feedback record to `eval/data/feedback.jsonl`.
   - Wire a thumbs-down button in the frontend chat UI.

3. **Add a `promote` CLI command** (`eval/promote.py`, new file):
   - `uv run python -m eval.promote --trace-id <id>` reads the trace from
     `traces.jsonl`, formats it as a golden item, and appends it to
     `eval/data/golden.json`.
   - Interactive mode: `uv run python -m eval.promote --interactive` lists
     recent traces (newest first, bad-rated first) and lets you select, review,
     add `expected_answer` and `key_facts`, then append.
   - Deduplicate by question text (warn if similar question already exists).

4. **Add a `list-traces` CLI command** that prints a table of recent traces
   with columns: `id | date | question (truncated) | rating | promoted`.

**Estimated effort:** 4–6 days  
**Files touched:** new `api/trace_log.py`, `api/main.py`, new
`eval/promote.py`, `frontend/` (thumbs-down button), `eval/data/.gitignore`

---

## Phase 5 — Prompt Experimentation

**Goal:** Run eval against two different prompt or retrieval configurations
side-by-side and generate a comparison report, so you can measure the real
impact of a prompt change before committing it.

**Why it matters:** Right now you can only compare prompts by running eval
twice, manually diffing the Markdown reports, and trying to remember what
changed. This phase makes A/B comparison first-class.

### Tasks

1. **Extract prompts into versioned config files** (`prompts/`, new directory):
   - Move the system prompts from `graph/chains/generation.py`,
     `graph/chains/generation_grader.py`, etc. into `.txt` or `.yaml` files
     under `prompts/`.
   - Each chain reads its prompt from the config file at startup (path
     configurable via env var or constructor arg).
   - Add a `prompts/default/` variant that mirrors the current prompts exactly,
     so nothing changes in behaviour until you create a new variant.

2. **Add `--prompt-variant` flag** to `eval/run_eval.py`:
   - `--prompt-variant prompts/experiment-1/` loads prompts from that directory
     instead of the default.
   - The eval report records which variant was used.

3. **Write a comparison script** (`eval/compare_variants.py`):
   - Takes two report JSON files as arguments.
   - Outputs a Markdown table: per-metric delta (A vs B), per-question winner,
     and overall recommendation.
   - Example: `uv run python -m eval.compare_variants report_A.json report_B.json`

4. **Add a convenience shell script** (`scripts/experiment.sh`):
   - Runs eval twice (once with `--prompt-variant A`, once with `B`) and then
     calls `compare_variants.py` automatically.

**Estimated effort:** 3–4 days  
**Files touched:** new `prompts/` directory, `graph/chains/*.py`,
`eval/run_eval.py`, new `eval/compare_variants.py`, new
`scripts/experiment.sh`

---

## Phase 6 — Human Review Queue

**Goal:** Provide a simple terminal UI for domain experts (radiation safety
specialists) to review flagged answers, approve or correct them, and optionally
write corrections back to the golden dataset.

**Why it matters:** LLM-as-judge metrics can miss domain-specific errors. A
medical or regulatory expert can catch issues that faithfulness and relevance
scores miss. This phase adds a lightweight human-in-the-loop layer without
requiring a full web UI.

### Tasks

1. **Auto-flag items in eval reports**:
   - Mark a question as `flagged: true` in the JSON report when any metric
     score is below a configurable `--flag-threshold` (default 0.6, higher than
     the 0.5 pass threshold).
   - Add a "Flagged for review" section to the Markdown report.

2. **Write a review CLI** (`eval/review.py`, new file):
   - `uv run python -m eval.review --report eval/reports/report_<ts>.json`
   - Iterates through flagged items one at a time in the terminal.
   - For each item, displays: question, retrieved context (truncated), generated
     answer, metric scores.
   - Prompts the reviewer: `[A]pprove / [R]eject / [C]orrect / [S]kip`
   - Records decisions to `eval/data/reviews.jsonl`.

3. **Support corrections**:
   - `[C]orrect` opens `$EDITOR` (or a simple inline prompt) for the reviewer
     to type the correct answer and optional key facts.
   - Corrections are saved to `reviews.jsonl` and can be promoted to
     `golden.json` via `eval.promote --from-reviews`.

4. **Review summary report**:
   - `uv run python -m eval.review --summary` prints a table of all reviewed
     items: question, decision, reviewer comment, date.
   - Useful for tracking how often the LLM metrics agree with human judgment
     (calibration check).

5. **Optional — web UI** (future, not in this phase):
   - If the terminal workflow proves insufficient, the same `reviews.jsonl`
     format can back a simple FastAPI + htmx review page added to the existing
     backend. The data format is designed for this upgrade path.

**Estimated effort:** 4–6 days  
**Files touched:** `eval/run_eval.py`, new `eval/review.py`, new
`eval/data/reviews.jsonl` schema doc

---

## Implementation Order

```
Phase 1 — Continuous Scoring          (1–2 days)   ← start here, biggest signal/effort ratio
Phase 3 — CI/CD Regression Gating    (2–3 days)   ← protects against regressions early
Phase 2 — Observability               (3–5 days)
Phase 5 — Prompt Experimentation      (3–4 days)
Phase 4 — Trace Capture               (4–6 days)
Phase 6 — Human Review Queue          (4–6 days)   ← most complex, least urgent
```

Phases 1 and 3 are independent and can be worked on in parallel.
Phases 4 and 6 share the `reviews.jsonl` / `golden.json` pipeline and should
be developed together or in sequence.

---

## Feature Parity Summary

| Braintrust Feature              | Phase | Status  |
|---------------------------------|-------|---------|
| Continuous 0–1 scoring          | 1     | planned |
| Score distributions in reports  | 1     | planned |
| Cost tracking per run           | 2     | planned |
| Latency tracking per node       | 2     | planned |
| CI/CD eval on every PR          | 3     | planned |
| Regression gating (block merge) | 3     | planned |
| PR comment with metric diff     | 3     | planned |
| Production trace logging        | 4     | planned |
| One-command dataset promotion   | 4     | planned |
| User feedback (thumbs down)     | 4     | planned |
| Prompt versioning               | 5     | planned |
| Side-by-side prompt comparison  | 5     | planned |
| Human review queue              | 6     | planned |
| Corrections → golden dataset    | 6     | planned |
