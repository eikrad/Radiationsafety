# Handoff: validate and run the held-out test set (#167)

For a new session that continues the work on `eval/data/golden_test.json`. Read this file first; it stands alone. Written 2026-10-13 on branch `ccr-833fa05d-4dad6i` (no pull request yet).

**Paste this to start the session:**

> Read `docs/handoff-held-out-test-run.md` in this repository and carry out steps 2 to 5. Stop and ask me before anything listed under "Ask before".

## 1. Why we are doing this

- The 39 questions in `eval/data/golden.json` are now a **development set**. The generation prompt (#145), the chunking of Danish law, `k = 5` and the embedder were all chosen by their results on exactly these questions, so the 38/39 of run `prompt-145` is a development number, not a measured generalisation (#167).
- 39 questions are also **too few to compare answer models or settings**: one question moves the pass rate by 2.6 points, and most questions pass for every model. Earlier model comparisons differed by one or two questions.
- #146 (retrieval experiments) and #168 (is the weak Danish-law retrieval caused by language or by structure?) both wait for **new questions written before any change**, so that a change is judged on questions it was not designed on.
- #150 (scope check) needs more questions the system should refuse, and answerable questions that merely sound out of scope.

So we wrote a second, **held-out** set from the source documents (not from system failures, not looking at any system answer), had every item checked against the source text by independent model reviewers, and approved all of it by hand.

## 2. What exists now

| | |
|---|---|
| File | `eval/data/golden_test.json`, 106 items, 396 nuggets |
| Mix | 100 to answer, 6 to refuse; 70 `dk-law`, 31 `iaea`, 5 `both`; 59 Danish, 47 English questions |
| Tiers | `tier: core` (50) and `tier: extended` (56) on every item; `difficulty` (`easy`/`medium`/`hard`) is the reviewers' rating, informational only |
| Runner | `uv run python -m eval.run_eval --golden eval/data/golden_test.json --tier core` (or `extended`, or no `--tier` for all 106) |
| Tests | `tests/test_golden_test_set.py` (tiers partition the set, refusals are in core, no easy answerable item in core, no id clash with the dev set, **no pointer to the source in a question**) |
| Docs | `eval/README.md`, section "Held-out test questions" and "Core and extended tiers" |
| Review status | **All 106 questions approved by the maintainer** in the review page. Nothing is pending. Do not re-review or reword questions without a reason from a run |

Properties of the questions that matter for reading results:

- **No question says where the answer is written** (no paragraph, annex, table, section, standard number or order name). Only the legal system is named, and only where Danish law and the IAEA standards can differ. This was a deliberate correction: a real user does not know the address of an answer.
- **Core** = all hard items, all 6 refusals, and medium items up to 50 (rarest topics first, then most vital nuggets). No answerable item rated easy. **Extended** = the rest (easy items, most Danish/English counterparts, leftover mediums). The tiers were fixed from the reviewers' ratings and topic counts **before any system ran**.
- Seven Danish/English pairs on the same passage (#168) are spread over the tiers: compare a pair only on the full run.
- Emergency (5 items, 3 in core) and research (6 items, 4 in core) are the thinnest topics.
- Evidence quotes were verified against the *extracted text of the source files*, **not yet against the real search index**. That is step 2.

## 3. Prerequisites

This must run on a machine that has the project environment: `uv sync`, a `.env` with `SCW_SECRET_KEY`, a populated `.chroma/` index built by `ingestion.py` (the cloud container that wrote the set had neither the index nor the dependencies; `torch` could not even be installed there). Check first:

```bash
git fetch origin ccr-833fa05d-4dad6i && git checkout ccr-833fa05d-4dad6i
uv sync
uv run pytest tests/test_golden_test_set.py tests/test_eval_golden.py -v
uv run pre-commit run --all-files        # black and isort were NOT run on the new code; fix what they change
uv run ruff check eval tests
```

The new Python code is small: `select_tier` in `eval/golden.py`, the `--tier` option in `eval/run_eval.py`, and `tests/test_golden_test_set.py`. Commit any formatting fix as `style:`.

The index must be the one the dev runs used. The last dev run (`20261011_061712`, label `prompt-145`) recorded the index fingerprint in `config.index` of `eval/history/runs.jsonl`; if the current index differs (re-ingestion, re-chunking, new documents such as #169), say so in the report, because evidence quotes depend on the chunks.

## 4. Steps

### Step 1: review (done)

All 106 questions were approved. There is nothing to apply. Do not change question text.

### Step 2: check the evidence against the real index

```bash
uv run python -m eval.golden --golden eval/data/golden_test.json --check-index
```

A vital nugget none of whose quotes is in any chunk is an **error** (it can never be found, so every system scores a miss). A quote in more than three chunks is a **warning** (too unspecific). Expected trouble spots: quotes taken from tables (annex rows, reference-level tables, the security-group and control-level tables, the SSR-6 category table, the GSR Part 3 emergency-worker table), which depend on how the chunker cut the table, and quotes that cross a chunk boundary.

How to fix, in this order of preference:
1. Add a second, **alternative** quote for the same nugget from the same passage (several quotes are alternatives; any one counts).
2. Shorten or re-anchor the quote on the distinguishing words, still verbatim as ingested, at most 150 characters.
3. Only if the passage is split across chunks in every cut: use the part that carries the fact.

Rules: change **evidence only**. Never change a nugget's meaning, importance, a question or an expected answer in this step. Re-run the check until there are no errors; list remaining warnings in the report. Commit as `fix(eval): re-anchor held-out evidence quotes on the real index`. Evidence changes alter `content_hash` but not `questions_hash`, so tiers stay comparable.

Why this comes first: a wrong quote is scored as a retrieval miss, which would blame the system for our labelling error.

### Step 3: baseline on the core tier

Use the configuration the dev set ended with, so the only new thing is the question set. Check `.env.example` and the last dev run for the exact values; the defaults are `LLM_PROVIDER=scaleway`, `SCW_MODEL=gemma-4-26b-a4b-it`, `EMBEDDING_PROVIDER=scaleway`, `SCW_EMBED_MODEL=bge-multilingual-gemma2`, `RETRIEVER_K=5`.

```bash
# judge: a different model than the answering one, 3 votes (same as the dev runs)
EVAL_GRADER_PROVIDER=scaleway EVAL_JUDGE_MODEL=qwen3.8-27b EVAL_JUDGE_VOTES=3 \
uv run python -m eval.run_eval --golden eval/data/golden_test.json --tier core \
  --no-web-search --label test-core-baseline \
  --notes "held-out core tier, config as prompt-145, no web search"
```

Decisions already made, with the reason:
- `--no-web-search`: web results are not in the corpus, they make runs irreproducible, and with web search on the system could "answer" a question we want it to refuse. **Caveat:** the dev run `prompt-145` was recorded with web search **on** (`config.web_search: true`), so its 38/39 and this run are not like for like. Say so when you compare them. If you want the comparison, run the dev set once more with `--no-web-search`.
- The same judge as the dev runs (`qwen3.8-27b`, 3 votes), then a second judge for robustness (below).
- A clean working tree. Commit before the run, otherwise the history entry is marked dirty (`--allow-dirty` only for a throwaway run).
- Allow time: the runner waits 5 s after each graph run and 20 s between items, plus generation, grading and judge calls. Plan about 40 to 70 minutes for 50 items (estimate, not measured). Do not lower the delays to save time unless the provider limits allow it.

Then re-score the same answers with a second judge, which costs only the judge calls:

```bash
EVAL_GRADER_PROVIDER=scaleway EVAL_JUDGE_MODEL=mistral-medium-3.5-128b \
uv run python -m eval.run_eval --rescore <RUN_ID> --label test-core-baseline-judge-mistral
```

Optionally validate a judge first with `uv run python -m eval.judge_check --judge scaleway:<model-a> --judge scaleway:<model-b>`.

### Step 4: confirmation on the extended tier

Same configuration, same judge, same flags:

```bash
EVAL_GRADER_PROVIDER=scaleway EVAL_JUDGE_MODEL=qwen3.8-27b EVAL_JUDGE_VOTES=3 \
uv run python -m eval.run_eval --golden eval/data/golden_test.json --tier extended \
  --no-web-search --label test-extended-baseline \
  --notes "held-out extended tier, config as test-core-baseline"
```

Run it **once**. `extended` is the untouched confirmation set: nothing is tuned on it. Optionally a third run without `--tier` gives the 106-question number and lets the seven Danish/English pairs be compared.

Report core and extended **side by side**, not averaged. A lower number than the dev set is a result, not a reason to tune on the test set.

### Step 5: read the results, then decide what to try

1. `uv run python -m eval.dashboard`; open `eval/reports/dashboard.html`. Read the headline split into "why questions failed" (retrieval vs generation), the per-topic and per-source breakdown and the question grid. Compare only runs with the same `questions_hash` (core with core).
2. Classify every failed question as one of: **retrieval miss** (evidence not in the context), **generation fault** (evidence there, answer wrong or unsupported), **label problem** (the system's answer is right, or the nugget or evidence is wrong or too strict), **refusal fault** (refused an answerable one, or answered one to refuse). Label problems are fixed in the golden file (evidence or nugget wording), recorded in the report, and the affected runs are re-scored with `--rescore`; they are not tuned around.
3. Look at the questions every configuration will pass or fail: if almost everything passes, say so (the set is too easy to separate anything) and plan new questions, **do not delete or re-tier these**.
4. Only then propose experiments, in this order of value, each with an **adoption rule written before measuring** (as the ROADMAP does):
   - the retrieval candidates parked in #146 (Danish translation of the question as a second query, smaller units in definition lists, re-ranking), now that questions exist that they were not designed on;
   - the language-versus-structure comparisons in #168, using the seven Danish/English pairs and retrieval-only runs (`--retrieval-only`);
   - the scope check in #150, measured on the six refusals plus the two borderline in-scope questions (aircrew, building materials) that must not be refused.
   Tune on **core only**. Any item used to design or choose a change counts as a development question from then on; confirm on `extended`. A confirmation after tuning on `extended` needs new questions.
5. Write the result into `ROADMAP.md` under a new dated section (what was measured, the numbers for core and extended, what can and cannot be claimed) and comment the key numbers on #167. Open follow-up issues for concrete findings.

## 5. Rules

- Do **not** tune on `extended`. Do **not** move items between tiers after any run (that would tune the set on its own results). Re-rate difficulty from results only to plan new questions.
- Do **not** reword questions to make them pass. A question that is genuinely ambiguous is a finding: report it, fix it in the file, re-score.
- Do **not** add pointers to the source (paragraph, annex, table, standard number, order name) to a question; the test rejects them.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`). Never edit version numbers by hand.
- Open pull requests against **`staging`**, never `master`, and only when the maintainer asks. Feature PRs are merged with Rebase and merge (CONTRIBUTING.md, CLAUDE.md).
- `eval/history/runs.jsonl` is committed and must contain the runs you report.

## 6. Known soft spots in the labels (check these before blaming the system)

- `test-both-transport-monitoring-dk-vs-iaea`: the IAEA guide SSG-86 also recommends individual monitoring at 1 to 6 mSv, and Danish law allows a dose-monitoring programme instead of a personal dosimeter, so the contrast is softer than the question suggests.
- `test-iaea-gsr6-decommissioning-plan-timing`: SSG-40 says the initial plan is prepared at the design stage, while the nugget says it goes in with the application to operate. Not yet reconciled.
- `test-both-radon-dk-vs-iaea` asks for dwellings; the IAEA also sets 1000 Bq/m3 for workplaces.
- Refusals with partial information in the corpus: `test-refuse-skift-p-nummer` (the order says register changes must be reported, without a deadline), `test-refuse-sst-id` (the order mentions an "ID" in the dose register, which is not an SST-id). A hedged answer that cites the general rule and says the specific point is not given should count as a correct refusal; if the judge does not, that is a judge issue to record.
- `test-dk-lukket-kilde-anbefalet-anvendelsesperiode` is an answer item, not a refusal: no maximum number of years is set, but the source certificate must state a recommended period of use.
- Table-row evidence (reference levels, security groups, control levels): see step 2.
- Difficulty ratings come from reviewing models, not from systems; `easy` items are in `extended` by design.

## 7. Ask before

- Changing any question, expected answer or nugget importance.
- Re-tiering items, or deleting any.
- Cleaning the five dev questions (`golden.json`) that still point to the source: it would make earlier runs incomparable. Decide with the maintainer.
- Opening a pull request, or merging.
- Re-ingesting documents (changes the chunks and the evidence check; related #169).

## 8. Related

#167 (held-out questions, this work), #146 (retrieval candidates), #168 (language versus structure), #150 (scope check), #133 (labelling), #169 (SST documents for the corpus), #170 (chat report export, separate).
