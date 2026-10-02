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
| `--regrade RUN_ID` | Run today's sufficiency grader on a saved run's first retrievals, plus each retrieval without its evidence (see [Checking the sufficiency grader](#checking-the-sufficiency-grader)); no answers, no judge |
| `--rescore RUN_ID` | Judge and score a saved run again (see [Re-scoring](#re-scoring)) without running the graph |
| `--retrieval-only` | Score only retrieval (see [Comparing retrieval settings](#comparing-retrieval-settings)): no answer model, no judge |
| `--depth K` | Retrieval-only: chunks retrieved per collection for the rank metrics (default 20) |
| `--char-budget CHARS` | Retrieval-only: characters per collection for `evidence_recall_budget` (default 7500, about what k=3 passes on today) |
| `--allow-dirty` | Record the run although tracked files have uncommitted changes (it is then marked dirty) |

**Rate limits:** by default the runner waits **5 s** after each graph run and **20 s** between items so eval stays under typical free-tier limits. Set the env vars above or use `--delay-after-graph 0 --delay-between-items 0` to disable delays.

## Comparing retrieval settings

```bash
EMBEDDING_PROVIDER=scaleway SCW_EMBED_MODEL=qwen3-embedding-8b LLM_PROVIDER=scaleway \
  uv run python -m eval.run_eval --retrieval-only --label emb-qwen3
```

Retrieves for every answerable question and scores retrieval only: no answer model, no judge, so a comparison takes seconds, costs only the query embeddings and carries no judge noise. There is no pass rate.

- `evidence_recall_initial` is the graph's own first retrieval (k=3 per collection, merged), the same metric as in a full run.
- A second, deeper retrieval (`--depth`, default 20 per collection) goes through the same retrievers. From that one ranked list come:
  - `evidence_recall_at_1/3/5/10/20`: share of vital nuggets whose evidence is in the top k of its collection;
  - `reciprocal_rank` per question (1/rank of the first chunk holding each vital nugget's evidence, 0 if not retrieved, averaged over the vital nuggets; its mean is MRR);
  - `evidence_recall_budget`: recall when each collection may pass on only `--char-budget` characters, chunks counted in rank order. Chunkings with longer chunks retrieve more text at equal k and win for that reason alone; the budget compares them at equal text (Dense X, Chen et al. 2024).

  A `k` or budget experiment therefore needs no extra run. Ranks are per collection, because the generator gets every collection's list in full.
- A self-check compares the deep list's top k with the graph's retrieval and lists disagreeing questions (`top_k_mismatches`); there recall@3 would not describe the graph.

The run records the embedding model, whether questions carried the model's instruction (`EMBED_QUERY_INSTRUCTION=false` measures the instruction's effect without re-ingestion), `retrieval_depth`, `char_budget` and the search index fingerprint. A retrieval-only run refuses to start when `LLM_PROVIDER=ollama` would override `EMBEDDING_PROVIDER`.

**Comparing two retrieval-only runs** in the dashboard counts questions whose evidence ranked higher or lower (reciprocal rank), not pass/fail flips, with the same sign test. Ranks move far more often than recall@3 flips, so the test has more questions to work with on the same golden set.

Build the collections for a new embedding model first with `EMBEDDING_PROVIDER=… uv run python ingestion.py --reembed-from gemini`: it embeds the existing chunks, so every model is compared on identical chunks and the golden evidence quotes stay valid.

### Pooling: passages the golden set does not list yet

A variant can retrieve a valid passage that is not among a nugget's evidence quotes; it then scores a miss, and comparisons favour the retriever the labels were made with (BEIR, Thakur et al. 2021). After comparing variants:

```bash
uv run python -m eval.pool RUN_ID_A RUN_ID_B --k 5
```

writes `eval/reports/pool_<time>.md`: for questions where a run missed the evidence, the chunks in its top k that no quote covers, with the rank each run gave them. Read them, add the shortest distinguishing verbatim span of each valid passage to that nugget's `evidence`, commit, and re-score each run without retrieving again:

```bash
uv run python -m eval.run_eval --rescore RUN_ID_A
```

No LLM decides relevance here: LLM relevance labels agree with people on system rankings but miss relevant passages (Thomas et al. 2024), and a rejected passage would silently stay a miss.

## Environment

- **Retrieval**: the embeddings chosen by `EMBEDDING_PROVIDER` (default Gemini, `GOOGLE_API_KEY`); their collections must be built. `LLM_PROVIDER=ollama` always retrieves with local embeddings.
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
- **Jurisdiction**: the system answers from Danish law and IAEA standards together. When the two give different answers (e.g. dose constraints for carers, dose rates at a radiography barrier), name the source in the question ("According to the IAEA safety standards, …"); otherwise an answer from the other source is right too and is scored as wrong.
- **`expected_behavior: refuse`** marks questions the sources do not answer; the system should say so. These items have no nuggets.
- **Tags**: `topics` (one or more of `medical`, `industrial`, `research`, `transport`, `waste`, `emergency`, `occupational`, `general`), `language` of the question, `source` where the answer lives (`dk-law`, `iaea`, `both`). They power the dashboard's per-topic view and are left out of the dataset fingerprint, so retagging keeps runs comparable.

`run_eval` validates the file and lists every problem at once (unknown topic, vital nugget without evidence, refuse item with nuggets, the v1 `key_facts` format, …).

Check the evidence quotes against the chunks actually in the search index, after editing the golden set and after any re-chunking:

```bash
uv run python -m eval.golden --check-index
```

A vital nugget none of whose quotes is in any chunk is an error: its evidence can never be found (misquoted, or split across a chunk boundary). An alternative quote in no chunk, and a quote in more than three chunks (too unspecific to tell the right passage apart), are warnings.

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
| **Grader right without the evidence** (`grade_documents_ablation_correct`, `--regrade` only) | The same first retrieval with its evidence chunks removed is judged insufficient. |
| **Sufficiency grader right** | Whether `grade_documents` judged the first retrieval correctly, measured against the evidence (CRAG, Yan et al. 2024, found prompted relevance judges much weaker than they look). |
| **Evidence position in context** (`evidence_position`) | Which chunk of the generator context (1 = first) first holds each vital nugget's evidence, averaged over the nuggets found. Position can matter as much as presence (Lost in the Middle, Liu et al. 2024); it tells a position effect from a generator error. |
| **Answers with a warning** (`warning_shown`) | 1 if the answer carried a warning for the user, e.g. "could not be fully verified". Lower is better. |

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

## Checking the sufficiency grader

```bash
uv run python -m eval.run_eval --regrade RUN_ID --label grader-v2
```

Runs `grade_documents` as the graph does, on the first retrieval saved with a full run (`outputs_<RUN_ID>.json`), without answering or judging:

- `grade_documents_correct`: the verdict is right when it says "sufficient" exactly if every vital nugget's evidence was retrieved, and "insufficient" for questions to refuse. The same metric as in a full run, on identical retrievals.
- `grade_documents_ablation_correct`: for each retrieval that held all its evidence, the chunks holding it are removed and the grader must now say "insufficient". This gives about as many labelled insufficient cases as there are answerable questions, without new retrieval (#129).

It costs about two grader calls per question, so a prompt change to the grader can be measured in minutes rather than with a full run.

## Position test

Does the generator use evidence less when it sits in the middle of the context
(#132)? `eval.position_test` answers each answerable question whose first
retrieval holds vital evidence twice from the same chunks, the evidence first
and the evidence in the middle, and compares lenient vital recall with a sign
test. One groundedness vote per answer; about 6 calls per question.

```bash
uv run python -m eval.position_test --label position-k3
```

Each run is appended to `eval/history/position_tests.jsonl` (commit it like
`runs.jsonl`).

## k test

Do more chunks per collection give better answers? `eval.k_test` answers every
golden question from the top 3 and from the top 5 chunks of each collection and
compares the scores per group: evidence already at k=3 (extra chunks can only
distract), evidence only at k=5 (where they can help), evidence missing, and
questions to refuse (refusal without unsupported claims scores 1). About 6
calls per question; runs go to `eval/history/k_tests.jsonl`.

```bash
uv run python -m eval.k_test --label k3-vs-k5
```

`RETRIEVER_K` sets k for the graph and for full runs.

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

Retrieval-only runs are re-scored from their saved ranked lists, without retrieving or judging (see [Pooling](#pooling-passages-the-golden-set-does-not-list-yet)). For full runs: judges and scores a saved run again with the current golden set, judge and scoring, without running the graph: change a nugget, the judge prompt or the judge model and see the effect for the cost of the judge calls only. The new history entry keeps the original run's git state and answer-side settings (the answers are the original ones), records the new judge and `rescored_from`, and is labelled `rescore of <run_id>` unless `--label` is given. Questions added to the golden set after the run are skipped with a warning.

## Run history

Every finished run records what it tested, so a score change can be traced to a cause:

| Field | Contents |
|--------|--------|
| `git` | commit, branch, and whether tracked files had uncommitted changes (captured at the start of the run; the run history and the documents ingestion rewrites, `documents/` and `document_versions.json`, do not count: the index fingerprint records what was indexed) |
| `config` | answer, judge and embedding models; whether the judge is the answering model; `retriever_k`; web search; `metrics_version`; a fingerprint of each `graph/chains/*.py` prompt module; `index`: per collection the number of chunks and a hash of their texts (independent of chunk ids, so re-embedded identical chunks match and any re-chunking shows) |
| `dataset` | `questions_hash` (ids + questions: runs with the same value are comparable), `content_hash` (also expected answers, expected behaviour and nuggets with their evidence quotes: changes when grading targets change, e.g. a pooled quote is added), `n_items` |
| `results` | per question: pass, error type, metrics, tags; generated answers stay in the local report |

A `--limit` run gets its own `questions_hash`, so it never mixes into full-set trends. A run that crashes records nothing. A run on uncommitted tracked files is not recorded unless `--allow-dirty` is given, since its commit would not say what ran; `--no-history` debug runs are always allowed.

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
