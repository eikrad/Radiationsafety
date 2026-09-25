# Evaluation harness

Systematic evaluation for the radiation safety RAG: a golden set of questions with the facts a good answer needs and where they are written in the sources, an independent LLM judge, deterministic scoring, a run history and a local dashboard.

## How to run

From the project root:

```bash
uv run python -m eval.run_eval --no-web-search --label baseline
```

The harness uses your `.env` for the LLMs (no API keys in golden data). Ensure ingestion has been run so the graph has documents to retrieve.

### Options

| Option | Description |
|--------|-------------|
| `--golden PATH` | Path to golden JSON (default: `eval/data/golden.json`) |
| `--limit N` | Run only on the first N items (useful for development) |
| `--no-web-search` | Disable web search for reproducible eval runs |
| `--output-dir PATH` | Directory for reports and saved graph outputs (default: `eval/reports`) |
| `--delay-after-graph SEC` | Seconds to wait after each graph run (default 5). Overrides `EVAL_DELAY_AFTER_GRAPH_SEC`. Use `0` to disable. |
| `--delay-between-items SEC` | Seconds to wait between golden items (default 20). Overrides `EVAL_DELAY_BETWEEN_ITEMS_SEC`. Use `0` to disable. |
| `--label TEXT` | Short name for the run in the history and dashboard, e.g. `baseline` or `dk-query-translation`. The latest run labelled `baseline` is what other runs are compared against. |
| `--notes TEXT` | Free-text notes on what the run tests |
| `--history-file PATH` | Run history to append to (default: `eval/history/runs.jsonl`) |
| `--no-history` | Do not record the run (e.g. quick debugging runs) |
| `--rescore RUN_ID` | Judge and score a saved run again (see [Re-scoring](#re-scoring)) without running the graph |

**Rate limits:** by default the runner waits **5 s** after each graph run and **20 s** between items so eval stays under typical free-tier limits. Set the env vars above or use `--delay-after-graph 0 --delay-between-items 0` to disable delays.

## Environment

- **Retrieval**: the graph uses **Gemini embeddings**; set **`GOOGLE_API_KEY`** in `.env` and run ingestion once.
- **Answers**: the graph uses `LLM_PROVIDER` and its key (`gemini`, `openai`, `mistral`, `scaleway`, `ollama`).
- **Judge**: `EVAL_GRADER_PROVIDER` picks the judge's provider and `EVAL_JUDGE_MODEL` its model. For Scaleway both are needed, e.g. `EVAL_GRADER_PROVIDER=scaleway` and `EVAL_JUDGE_MODEL=<model id>` (plus `SCW_SECRET_KEY`). **Use a different model than the one that writes the answers**: a model judging its own answers is lenient. Without either variable the answering model judges; the run records `judge_is_generator: true` and prints a warning. Check a judge with [`judge_check`](#checking-the-judge) before trusting it.
- **Eval delays** (optional): `EVAL_DELAY_AFTER_GRAPH_SEC`, `EVAL_DELAY_BETWEEN_ITEMS_SEC`.
- **Optional – LangSmith**: see [LangSmith](#langsmith).

## Golden set

`eval/data/golden.json` is a list of items:

```json
{"id": "bek-stråling-dosisgrænser",
 "question": "Hvor findes dosisgrænserne for erhvervsmæssig bestråling?",
 "topics": ["occupational"], "language": "da", "source": "dk-law",
 "expected_behavior": "answer",
 "expected_answer": "…reference answer for reviewers…",
 "nuggets": [
   {"text": "Dosisgrænserne for erhvervsmæssig bestråling fremgår af bilag 2",
    "importance": "vital",
    "evidence": ["for erhvervsmæssig bestråling gælder de dosisgrænser for effektiv dosis og ækvivalent dosis, der fremgår af bilag 2"]},
   {"text": "Grænserne gælder både for effektiv dosis og ækvivalent dosis",
    "importance": "okay", "evidence": ["…"]}]}
```

- **Nuggets** are atomic facts a good answer contains (about one short sentence each, a yes/no decision for the judge). `vital` = the answer is wrong without it; `okay` = worth having. At least one vital nugget per answerable question. Nugget text can be in any language; the judge assigns by meaning.
- **Evidence** quotes are copied **verbatim** from the source as ingested (the document's own language). Several quotes are *alternatives* (the same fact in the PDF and the XML version, or in a guidance document); any one counts. Keep quotes short (≤ 150 characters, the validator warns otherwise) and specific: sister regulations often share wording, so include the part that distinguishes the right paragraph. Choose passages by reading the regulation, not by keyword search, and add another quote when a retrieval variant finds a different valid passage (BEIR, Thakur et al. 2021, shows how keyword-chosen labels bias comparisons toward keyword retrieval).
- **`expected_behavior: refuse`** marks questions the sources do not answer; the system should say so. These items have no nuggets.
- **Tags**: `topics` (one or more of `medical`, `industrial`, `research`, `transport`, `waste`, `emergency`, `occupational`, `general`), `language` of the question, `source` where the answer lives (`dk-law`, `iaea`, `both`). They power the dashboard's per-topic view and are left out of the dataset fingerprint, so retagging keeps runs comparable.

`run_eval` validates the file and lists every problem at once (unknown topic, vital nugget without evidence, refuse item with nuggets, the v1 `key_facts` format, …).

## Scoring

Per question the judge answers two narrow questions, and everything else is derived without tokens (`eval/scoring.py`):

1. **Nugget assignment** (answer + nuggets, no retrieved context, ≤ 10 nuggets per call): is each nugget *supported*, *partially supported* or *not supported* by the answer?
2. **Groundedness** (answer + the context the generator saw): which claims does the context not support, and did the answer refuse?

The groundedness question is asked up to `EVAL_JUDGE_VOTES` times (default 3) and the majority decides, separately for "has an unsupported claim" and "refused"; asking stops once two votes agree, so it costs about two calls per question. A tie (after a failed vote) counts as flagged, like the strict pass rule. Why: in the first Scaleway baseline, judging the same answers three times at temperature 0 with a single vote gave pass rates of 54 %, 62 % and 46 %. The report keeps each verdict's vote count and flagged claims.

Two focused calls rather than one combined prompt: in GroUSE (Muller et al. 2024), Llama-3 8B passed 69 % of the judge unit tests with separate calls per metric but 40 % with all metrics in one prompt.

| Metric | Meaning |
|--------|--------|
| **Pass** | Answerable: every vital nugget fully supported and no unsupported claim. Refuse item: the answer refused and made no unsupported claim. |
| **Vital facts in answer** (`vital_recall`) | Share of vital nuggets fully supported (strict, as V_strict in Pradeep et al. 2025). `vital_recall_lenient` counts partial support as ½. |
| **All facts in answer** (`all_recall`) | Share of all nuggets fully supported. |
| **Grounded vital facts** (`grounded_vital_recall`) | Vital nuggets that are in the answer *and* whose evidence was in the context. Facts the model knows from pretraining do not count (Trust-Score, Song et al. 2025); this matters where Danish rules differ from IAEA defaults. |
| **Retrieved facts used** (`context_utilization`) | Of the vital facts whose evidence reached the generator, the share the answer used (RAGChecker, Ru et al. 2024). |
| **Evidence in 1st retrieval / in context** | Share of vital nuggets whose evidence quote is in the first retrieval / in the generator's context (after `retrieve_missing`). Deterministic, no tokens. |
| **Answers with unsupported claim** | 1 if the answer contains any claim the context does not support. Lower is better. |
| **Sufficiency grader right** | Whether `grade_documents` judged the first retrieval correctly, measured against the evidence (CRAG, Yan et al. 2024, found prompted relevance judges much weaker than they look). |

Metrics that do not apply (e.g. vital recall of a refuse item) are left out rather than counted as 0. Each question also gets an **error type** that says whether retrieval or generation failed:

| `error_type` | Meaning |
|---|---|
| `ok` | passed |
| `retrieval_miss` | a vital fact's evidence never reached the generator; refusing is then even correct behaviour |
| `unsupported_claim` | evidence was there, but the answer claims something the context does not support |
| `wrong_refusal` | evidence was there, but the answer refused |
| `generator_miss` | evidence was there, but a vital fact is missing from the answer |
| `missed_refusal` | a question the sources do not answer was answered anyway |
| `judge_error` | the judge failed twice; the question is left out of the pass rate |

Matching evidence quotes ignores case, whitespace, soft hyphens, zero-width characters (present in the Danish XML) and typographic dashes.

## Checking the judge

```bash
uv run python -m eval.judge_check --judge scaleway:<model-a> --judge scaleway:<model-b>
```

Runs each judge over `eval/judge_fixtures.json`: 16 cases with the verdict a good judge must reach (correct and partial answers, fabrications, correct and wrong refusals, answering from own knowledge, an absurd context the judge must follow rather than its own knowledge, Gy vs Sv, µSv vs mSv, wrong annex, a foreign rule instead of the Danish one, translation, paraphrase). Prints the pass count and what each judge got wrong. Run it when the judge prompt or model changes; it costs about two calls per fixture and is not part of CI.

## Output

- **JSON**: `eval/reports/report_<run_id>.json` – run header, summary and per-question results.
- **Markdown**: `eval/reports/report_<run_id>.md` – the same for reading, with the error type and node path per question.
- **Graph outputs**: `eval/reports/outputs_<run_id>.json` – full answers, first retrieval, generator context, sufficiency verdict, node path; used for re-scoring.
- **History**: one line per run appended to `eval/history/runs.jsonl` (committed).
- **Dashboard**: `eval/reports/dashboard.html`, refreshed after every recorded run.

Reports, outputs and the dashboard are gitignored; the history is committed so runs stay comparable across machines and over time.

## Re-scoring

```bash
uv run python -m eval.run_eval --rescore 20260926_101500
```

Judges and scores a saved run again with the current golden set, judge and scoring, without running the graph: change a nugget, the judge prompt or the judge model and see the effect for the cost of the judge calls only. The new history entry keeps the original run's git state and answer-side settings (the answers are the original ones), records the new judge and `rescored_from`, and is labelled `rescore of <run_id>` unless `--label` is given. Questions added to the golden set after the run are skipped with a warning.

## Run history

Every finished run records what it tested, so a score change can be traced to a cause:

| Field | Contents |
|--------|--------|
| `git` | commit, branch, and whether tracked files had uncommitted changes (captured at the start of the run) |
| `config` | answer, judge and embedding models; whether the judge is the answering model; `retriever_k`; web search; `metrics_version`; a fingerprint of each `graph/chains/*.py` prompt module |
| `dataset` | `questions_hash` (ids + questions: runs with the same value are comparable), `content_hash` (also expected answers and nuggets: changes when grading targets change), `n_items` |
| `results` | per question: pass, error type, metrics, tags; generated answers stay in the local report |

A `--limit` run gets its own `questions_hash`, so it never mixes into full-set trends. A run that crashes records nothing.

Backfill older reports (they have no header, so settings show as "not recorded"):

```bash
uv run python -m eval.history import-reports --since 20260916
```

When a metric's definition changes, bump `METRICS_VERSION` in `eval/scoring.py`; the dashboard then marks runs on either side as not directly comparable. The same happens when the judge changes (provider, model or number of votes; runs from before voting count as one vote). To compare old runs under a new judge, re-score them with `--rescore RUN_ID`.

## Dashboard

```bash
uv run python -m eval.dashboard --open
```

| Option | Description |
|--------|-------------|
| `--baseline RUN_ID` | Compare runs against this run instead of the latest run labelled `baseline` (else the question set's first run) |
| `--output PATH` | Where to write the page (default: `eval/reports/dashboard.html`) |
| `--history-file PATH` | History to read (default: `eval/history/runs.jsonl`) |
| `--reports-dir PATH` | Local reports, for answer previews in the comparison (default: `eval/reports`) |
| `--open` | Open the page in the browser |

One self-contained HTML file, no network needed. Pick a question set, a run, and whether to compare it with the baseline or the previous run; every section follows that choice:

- **Headline**: pass rate with change vs previous and baseline, the verdict (regressions / improvements) with a sign test, and **why questions failed** (retrieval vs generation).
- **Scores over time**: pass rate and each metric as small charts in run order; dashed lines mark where scoring or grading targets changed; the ringed point is the baseline.
- **Compare runs**: score changes, changed settings (models, `retriever_k`, prompts, …), questions that flipped (regressions first, with the reason) and answer previews, and caveats such as uncommitted changes. Runs scored under different rules are marked *not directly comparable* instead of getting a regression verdict.
- **By topic**: pass rate per topic, language or source (judged questions only).
- **Questions**: ✓ / ✗ / ? (not judged) for every question in every run, with the reason on hover.

Typical workflow for a retrieval or prompt change:

```bash
uv run python -m eval.run_eval --no-web-search --label baseline          # on staging
# …make the change on a feature branch…
uv run python -m eval.run_eval --no-web-search --label dk-query-translation
uv run python -m eval.dashboard --open
```

With a small golden set one question flipping moves the pass rate by several points; check which questions flipped before reading a trend. Every comparison therefore carries an exact two-sided **sign test** over the questions that flipped between pass and fail (regressions vs improvements; unchanged questions carry no information about direction). With few flips nothing is significant — 5 of 5 in one direction still gives p = 0.0625 — which is the intended reading: single flips are hints, not findings. Per-question judgements are also the least reliable part of LLM judging (Pradeep et al. 2025 found good agreement with human assessors per run, but weak agreement per question).

## LangSmith

To trace eval runs in LangSmith:

1. Set in `.env`: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=radiation-safety-rag` (or your project), and `LANGCHAIN_API_KEY=...` (from [LangSmith](https://smith.langchain.com)).
2. Run: `uv run python -m eval.run_eval`.
3. In the LangSmith UI, filter runs by tags **eval** and **golden** to see evaluation runs. Each graph run and judge call is traced so you can inspect retrieval, generation, and judging per question.
