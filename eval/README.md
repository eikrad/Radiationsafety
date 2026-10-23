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
| `--split tune\|confirm` | Run only the held-out items of one split (`--golden eval/data/golden_test.json`, see [Tune and confirm splits](#tune-and-confirm-splits)) |
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
| `--char-budget CHARS` | Retrieval-only: characters per collection for `evidence_recall_budget` (default 7500, about what k=3 passed on; k=5 passes more) |
| `--allow-dirty` | Record the run although tracked files have uncommitted changes (it is then marked dirty) |

**Rate limits:** by default the runner waits **5 s** after each graph run and **20 s** between items so eval stays under typical free-tier limits. Set the env vars above or use `--delay-after-graph 0 --delay-between-items 0` to disable delays.

## Comparing retrieval settings

```bash
EMBEDDING_PROVIDER=scaleway SCW_EMBED_MODEL=qwen3-embedding-8b LLM_PROVIDER=scaleway \
  uv run python -m eval.run_eval --retrieval-only --label emb-qwen3
```

Retrieves for every answerable question and scores retrieval only: no answer model, no judge, so a comparison takes seconds, costs only the query embeddings and carries no judge noise. There is no pass rate.

- `evidence_recall_initial` is the graph's own first retrieval (`RETRIEVER_K` per collection, default 5, merged), the same metric as in a full run.
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
- **No pointers to the source**: a question must not say where the answer is written (no paragraph, annex, table or section, no standard number such as `GSR Part 3` or `SSG-46`, no name of an order). A real user does not know that; it gives the retriever the answer's address and makes the question easier. Name only the legal system ("under Danish rules", "according to the IAEA safety standards") and only where the two give different answers. Describe the situation instead, so that its content tells the retriever which order applies. `tests/test_golden_test_set.py` checks this for the held-out set.
- **Jurisdiction**: the system answers from Danish law and IAEA standards together. When the two give different answers (e.g. dose constraints for carers, dose rates at a radiography barrier), name the source in the question ("According to the IAEA safety standards, …"); otherwise an answer from the other source is right too and is scored as wrong.
- **`expected_behavior: refuse`** marks questions the sources do not answer; the system should say so. These items have no nuggets.
- **Tags**: `topics` (one or more of `medical`, `industrial`, `research`, `transport`, `waste`, `emergency`, `occupational`, `general`), `language` of the question, `source` where the answer lives (`dk-law`, `iaea`, `both`). They power the dashboard's per-topic view and are left out of the dataset fingerprint, so retagging keeps runs comparable.

### Held-out test questions

`eval/data/golden_test.json` holds questions that are kept apart from the development set (#167). They were written from the source documents, before any system answer was looked at. Nothing is tuned on them: after a tuning step they count as development questions and a confirmation needs new ones. Run them with `--golden eval/data/golden_test.json`; their `questions_hash` then never mixes into the development trend.

They weight what separates models and settings: conditions where "eller"/"samt" decide the outcome (category A vs B, both conditions of a disposal rule), boundary values, computed cases (security group from an activity, a TI from a dose rate, a mixture sum), numbers and units in annex tables, definition lists, multi-hop references between a paragraph and an annex, questions that need two documents (a requirement and its guide), Danish rule vs IAEA value, English questions about Danish law and Danish/English pairs on the same passage (#168), borderline in-scope topics that sound out of scope (aircrew, building materials), and six questions to refuse (in-domain ones such as the SST-id and a deadline for a P-number change, which the sources do not give). Every item was checked against the source text by independent model reviewers, the findings were worked in, and the maintainer approved every item. Evidence quotes are verbatim as ingested: they were checked against the index the development runs used (2026-10-13), where tables are linearised as "row, column = value" and sub- and superscripts read `222 Rn`, `A_1`, `CTDI_vol`. Re-run `uv run python -m eval.golden --golden eval/data/golden_test.json --check-index` after any re-ingestion.

#### Tune and confirm splits

Every item in `golden_test.json` carries a `split` and the reviewers' `difficulty` rating (`easy`, `medium`, `hard`, informational only). Neither tag is part of the dataset fingerprint.

| Split | Items | Use |
|-------|-------|-----|
| `tune` | 53 | Tuning: retrieval, prompts, model choice. Failures are read and classified question by question. Holds every item whose evidence sits in a chunk the development set also quotes. |
| `confirm` | 53 | Untouched confirmation. Only aggregate numbers are read: pass rate with its interval, error types, flips against a baseline. |

```bash
uv run python -m eval.run_eval --golden eval/data/golden_test.json --split tune --label tune-baseline
uv run python -m eval.run_eval --golden eval/data/golden_test.json --split confirm --label confirm-baseline
```

A run on one split covers fewer items, so it has its own `questions_hash` and the dashboard never compares it with a run on the other split.

How the split was made (`eval/split.py`, `uv run python -m eval.split`), before any system ran on these questions:

- Items whose evidence sits in the same chunk of the search index form one group and share a split: tuning on one of them tunes the retrieval of the chunk the other tests. Chunks, not wording, decide, because sister regulations repeat phrases in different paragraphs. The Danish/English pairs (#168) therefore sit whole in one split, five in `tune` and two in `confirm`.
- A group with a chunk the development set quotes goes to `tune`, so `confirm` stays free of chunks that shaped earlier choices.
- The other groups are placed largest first, in an order fixed by a seed, on the side that keeps every stratum (expected behaviour × source × difficulty), language and topic most evenly divided. Result: 3 items to refuse and 6 hard items on each side.

Rules:

- Reading `confirm` failures question by question turns it into development data: ideas come from what was read, and label fixes made only where the system failed push the number up. Do that only at the end of a cycle, on purpose, and then write new confirm questions. A label problem found without looking at answers (e.g. by `--check-index`) may be fixed on both sides.
- Confirm a change on `confirm` at most three times per set of confirm questions, each time with the adoption rule committed before the run; after that, `confirm` counts as development data.
- Never move an item after a run. New questions are added without a split and `uv run python -m eval.split` places them: an item that shares a chunk with an assigned one joins its split, existing items keep theirs. New questions change the `questions_hash`, so baselines are run again.
- After re-ingestion or new documents (#169), `--check-index` also reports split leaks: a chunk shared across the splits, or a `confirm` item on a chunk the development set quotes. A leak is recorded with the results, not fixed by moving items. New documents can also answer a question to refuse: check those six again.
- The reviewers' difficulty ratings are model judgements made before the questions met a system. Re-rate from real results only to plan new questions.

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
| `graph_error` | the graph raised (after one more try if the connection dropped); counts as failed, the error is recorded, and the run goes on; if every question fails, the run stops unrecorded |

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

`RETRIEVER_K` sets k for the graph and for full runs (default 5, step 5 of the ROADMAP).

## Prompt test

Does the revised generation prompt answer better than the previous one (#145)?
`eval.prompt_test` replays the context each question's generator saw in a
saved full run (`eval/reports/outputs_<run_id>.json`, web results included) and
answers it with the prompt from before #145 (`PREVIOUS`, kept verbatim) and
with today's `RAG_SYSTEM`. Only the prompt differs. Both answers are scored as
in a full run, with 3 groundedness votes; the summary checks the adoption rule
from the ROADMAP. About 8–10 calls per question; runs go to
`eval/history/prompt_tests.jsonl`, the answers to
`eval/reports/prompt_test_<run_id>.json`. It refuses to run on uncommitted
code, since the commit has to say which prompt was tested.

```bash
uv run python -m eval.prompt_test --outputs eval/reports/outputs_20261003_132636.json --label prompt-145
```

After a judge change, `--rejudge eval/reports/prompt_test_<run_id>.json` judges
the saved answers of both prompts again instead of generating new ones (about
4–6 calls per question); `--outputs` must be the run whose contexts they
answered.

## Checking the judge

```bash
uv run python -m eval.judge_check --judge scaleway:<model-a> --judge scaleway:<model-b>
```

Runs each judge over `eval/judge_fixtures.json`: 16 cases with the verdict a good judge must reach (correct and partial answers, fabrications, correct and wrong refusals, answering from own knowledge, an absurd context the judge must follow rather than its own knowledge, Gy vs Sv, µSv vs mSv, wrong annex, a foreign rule instead of the Danish one, translation, paraphrase). Prints the pass count and what each judge got wrong. Run it when the judge prompt or model changes; it costs about two calls per fixture and is not part of CI.

## Output

- **JSON**: `eval/reports/report_<run_id>.json` – run header, summary and per-question results.
- **Markdown**: `eval/reports/report_<run_id>.md` – the same for reading, with the error type and node path per question.
- **Graph outputs**: `eval/reports/outputs_<run_id>.json` – full answers, first retrieval, generator context, sufficiency verdict, node path; used for re-scoring.
- **History**: one line per run appended to `eval/history/runs.jsonl` (committed), with the tokens each model used (`usage`) and the pass rate's 95 % interval.
- **Archived answers**: `eval/history/outputs/<run_id>.json.gz` (committed), a gzipped copy of the graph outputs of every recorded full run, about 200 KB per 50 questions. Retrieval-only runs are not archived: they are cheap to repeat. Commit the archive with the history line.
- **Dashboard**: `eval/reports/dashboard.html`, refreshed after every recorded run. It shows each run's cost at the prices valid on its day (`eval/data/prices.json`, per million tokens; add an entry when a price changes, do not edit the old one).

Reports, local outputs and the dashboard are gitignored; the history and the archived answers are committed, so runs stay comparable across machines and over time, and a run can be judged again after the model that answered it is gone.

## Re-scoring

```bash
uv run python -m eval.run_eval --rescore 20260926_101500
```

Retrieval-only runs are re-scored from their saved ranked lists, without retrieving or judging (see [Pooling](#pooling-passages-the-golden-set-does-not-list-yet)). For full runs: judges and scores a saved run again with the current golden set, judge and scoring, without running the graph: change a nugget, the judge prompt or the judge model and see the effect for the cost of the judge calls only. The new history entry keeps the original run's git state and answer-side settings (the answers are the original ones), records the new judge and `rescored_from`, and is labelled `rescore of <run_id>` unless `--label` is given. Questions added to the golden set after the run are skipped with a warning. Without the local `outputs_<run_id>.json` (another machine), the archived copy in `eval/history/outputs/` is used.

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

One self-contained HTML file, no network needed. Pick a question set (held-out `tune` and `confirm`, the dev set, older dev versions by date), then two runs in the **Runs** table: **B**, the run to look at, and **A**, the run to compare it with. Any two runs of a set can be compared either way; the bar at the top names A and B and swaps them. By default B is the newest run with new answers and A the baseline, or, when the baseline was scored differently or by another judge, the latest earlier run that was not. Every section follows that choice:

- **Headline**: B's pass rate with its 95 % interval and the count passed, the change against A, the verdict (regressions / improvements) with a sign test, and **why questions failed** (retrieval vs generation).
- **Runs**: every run, newest first, with kind (new answers, rescore by another judge, regrade, grader check, retrieval only; switch kinds on and off), answer model, judge, passed count, pass rate with its **95 % interval as a bar** on one shared scale, and cost. Bars that overlap a lot mean the difference may be chance.
- **Compare A → B**: score changes (pass rate with both intervals), changed settings (models, `retriever_k`, prompts, …), questions that flipped (regressions first, with the reason) and answer previews, and caveats such as uncommitted changes. Runs scored under different rules are marked *not directly comparable* instead of getting a regression verdict.
- **Scores over time**: pass rate (with its interval as a vertical bar) and each metric as small charts in run order; dashed lines mark where scoring or grading targets changed; the ringed point is the baseline.
- **By topic**: pass rate per topic, language or source (judged questions only).
- **Questions**: ✓ / ✗ / ? (not judged) for every question in every run with answers, A and B marked, the reason on hover. The header row and the question column stay in view while scrolling.

On a `confirm` set the page shows single questions like any other; the rule to read `confirm` in aggregate only still applies, so stay with the headline and the runs table there.

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
