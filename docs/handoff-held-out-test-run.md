# Handoff: run the held-out test set (#167)

For a new session that runs and evaluates `eval/data/golden_test.json`. Read this file first; it stands alone. Rewritten 2026-10-15 on branch `ccr-833fa05d-4dad6i` (no pull request yet). The rules for the split are in `eval/README.md`, section "Held-out test questions"; this file is the run plan.

**Paste this to start the session:**

> Read `docs/handoff-held-out-test-run.md` in this repository and carry out steps 1 to 4. Stop and ask me before anything listed under "Ask before", and before step 5.

## 1. Why

- The 39 questions in `eval/data/golden.json` are a **development set**. The generation prompt (#145), the chunking of Danish law, `k = 5` and the embedder were chosen on exactly these questions, so the 38/39 of run `prompt-145` is a development number, not a measured generalisation.
- 39 questions are too few to compare settings: one question moves the pass rate by 2.6 points.
- #146 (retrieval candidates), #168 (is weak Danish-law retrieval caused by language or by structure?) and #150 (scope check) wait for questions written before any change.

## 2. What exists now

| | |
|---|---|
| Questions | `eval/data/golden_test.json`: 106 items, 100 to answer, 6 to refuse; written from the source documents before any system answer was seen; every item approved by the maintainer |
| Split | `split: tune` (53) and `split: confirm` (53), made by `eval/split.py` before any run. Items whose evidence sits in the same index chunk share a split; every item on a chunk the development set quotes is in `tune`. 3 refusals and 6 hard items on each side; all Danish/English pairs whole in one split (5 tune, 2 confirm) |
| Labels | Every evidence quote is verbatim in the index the dev runs used (index fingerprint `b0fc52f61a4e` / `d90d785f314c`, 9436 + 944 chunks): `--check-index` reports 0 errors, 0 warnings, no split leaks. The known soft spots were resolved before any run (commit `328a341`) |
| Runner | `uv run python -m eval.run_eval --golden eval/data/golden_test.json --split tune` (or `confirm`) |
| Records | Each run records its split (`dataset.split`), the pass rate with its 95 % Wilson interval, and the tokens per model (`usage.answers`, `usage.judge`). Full runs archive their answers in `eval/history/outputs/<run_id>.json.gz` (commit them with the history); `--rescore` uses the archive when the local outputs are missing |
| Dashboard | `uv run python -m eval.dashboard`: split and number of new runs per confirm set, interval under the pass rate, cost per run from `eval/data/prices.json` |

## 3. Success criterion (fixed before the baseline)

The round succeeds when, at its end, these four things exist, whatever the numbers are:

1. Dev and test compared like for like (both without web search), each with its 95 % interval.
2. Every failed `tune` question classified: retrieval miss, generation fault, label problem or scope (refusal) fault.
3. Every experiment adopted or rejected by a rule committed before it was measured.
4. The gap between the optimised Scaleway configuration and Gemini 3.8 Flash measured on `confirm`.

A pass rate is not the target. Expectation recorded before measuring, a guess and no target: the test baseline lands clearly below the dev 38/39, roughly 65 to 85 %, because the set is harder (tables, computed cases, English questions on Danish law) and nothing was tuned on it. A lower number is a result, not a regression.

## 4. Decisions already made

- **Optimise only with the configured Scaleway models, only on `tune`.** Answer model `gemma-4-26b-a4b-it`, embedder `bge-multilingual-gemma2`, `k = 5`, no web search. `deepseek-v4-flash-0731` and `qwen3.8-27b` stay UI options; no answer baselines for them.
- **Judges.** Main judge `qwen3.8-27b`, 3 votes (as in the dev runs). Second judge `mistral-medium-3.5-128b` only at decision points: (a) one rescore of the first Gemma `tune` baseline, to see how far the judges agree on this set; (b) a rescore of every candidate that meets its adoption rule under Qwen, adopted only if the rule holds under both judges; (c) the final comparison on `confirm`.
- **Local embedder as a free counter-check, not for optimising.** Every retrieval experiment also runs retrieval-only with `bge-m3` (Ollama collections, same chunk texts and fingerprint as the Scaleway index, so split and evidence checks hold). Adoption is decided on `bge-multilingual-gemma2`; a change that helps only one embedder is reported as such. Privacy mode itself is not optimised in this round.
- **Frontier comparison only at the end, only on `confirm`**, with the final pipeline, against `gemini-3.8-flash`. Comparing on `tune` would favour Gemma, which the pipeline was tuned with.
- **Budget, lean plan:** about €12–18 at Scaleway (prices before tax) plus about $2–4 at Google. Retrieval-only runs cost almost nothing; a full `tune` run (Gemma + Qwen judge) about €0.75–1.15; a Mistral rescore of one split about €1.10. The dashboard shows the real cost of every run.

## 5. Prerequisites

On a machine with the project environment (`uv sync`), a `.env` with `SCW_SECRET_KEY` (and `GOOGLE_API_KEY` for step 5), the Scaleway index, and Ollama with `bge-m3` collections for the counter-check. Do not read or print `.env`.

```bash
git checkout ccr-833fa05d-4dad6i
uv sync
uv run pytest tests/ -q
uv run python -m eval.golden --golden eval/data/golden_test.json --check-index
```

The index check must report 0 errors and no split leaks. If the index fingerprint differs from `b0fc52f61a4e` / `d90d785f314c` (re-ingestion, new documents such as #169), stop and ask: evidence and split depend on the chunks.

Commit before every recorded run (a dirty tree is refused), and after it commit `eval/history/runs.jsonl` together with the new archive file in `eval/history/outputs/`.

## 6. Steps

The judge settings for every full run and rescore below:

```bash
export EVAL_GRADER_PROVIDER=scaleway EVAL_JUDGE_MODEL=qwen3.8-27b EVAL_JUDGE_VOTES=3
```

### Step 1: smoke test

```bash
uv run python -m eval.run_eval --golden eval/data/golden_test.json --split tune \
  --limit 2 --no-web-search --no-history
```

Check in the report and the console: answer model `gemma-4-26b-a4b-it`, embedder `bge-multilingual-gemma2`, `retriever_k` 5, `web_search` false, the index fingerprint above, and `usage` with tokens for the answer model and the judge. The model names in `usage` must match `eval/data/prices.json`; if a run's cost shows "no price for …", fix the price table (new entry), not the run.

### Step 2: baselines

1. **Dev set without web search**, for the like-for-like comparison (`prompt-145` ran with web search on; it was attempted on 4 of its 39 questions):
   ```bash
   uv run python -m eval.run_eval --no-web-search --label dev-no-web \
     --notes "dev set, config as prompt-145 but without web search"
   ```
2. **`tune` baseline, twice**, for the noise floor (questions that flip with nothing changed):
   ```bash
   uv run python -m eval.run_eval --golden eval/data/golden_test.json --split tune \
     --no-web-search --label tune-baseline --notes "held-out tune, config as prompt-145, no web"
   uv run python -m eval.run_eval --golden eval/data/golden_test.json --split tune \
     --no-web-search --label tune-baseline-repeat --notes "repeat for the noise floor"
   ```
3. **Second judge** on the first `tune` baseline:
   ```bash
   EVAL_JUDGE_MODEL=mistral-medium-3.5-128b uv run python -m eval.run_eval \
     --golden eval/data/golden_test.json --rescore <RUN_ID> --label tune-baseline-judge-mistral
   ```
4. **`confirm` baseline, once.** Read only the aggregate numbers (pass rate with interval, error-type counts):
   ```bash
   uv run python -m eval.run_eval --golden eval/data/golden_test.json --split confirm \
     --no-web-search --label confirm-baseline --notes "held-out confirm, config as tune-baseline"
   ```
5. **Retrieval-only baselines on `tune`**, with both embedders:
   ```bash
   uv run python -m eval.run_eval --golden eval/data/golden_test.json --split tune \
     --retrieval-only --label tune-retrieval-bge-gemma2
   EMBEDDING_PROVIDER=ollama uv run python -m eval.run_eval --golden eval/data/golden_test.json \
     --split tune --retrieval-only --label tune-retrieval-bge-m3
   ```

Allow about 35 to 45 minutes per full run of 53 questions (measured: 39 s per question for `prompt-145`; harder questions loop more). After the first full run, compare its cost in the dashboard with the estimate above; if a `tune` run costs more than about €1.50, stop and re-estimate the budget with the maintainer.

### Step 3: read the results

1. Headline: dev without web search vs `tune` + `confirm` baseline, each with its interval. The full test number is `tune` and `confirm` together (106 questions); also give it without the 14 questions on development chunks, which are all in `tune`.
2. Noise floor: how many questions flipped between the two `tune` baselines. Judge agreement: how many verdicts Mistral changed.
3. Classify every failed **`tune`** question (retrieval miss, generation fault, label problem, scope fault). Check first the hypotheses written before measuring: table chunks (linearised as "row, column = value"), Danish law asked in English, computed and multi-hop questions, refusal traps such as the 21-day transfer deadline for the P-number question.
4. **`confirm`: aggregate numbers only.** No question-by-question reading (eval/README.md).
5. Label problems found on `tune` are fixed in the golden file (evidence or nugget wording), recorded with the reason, and the affected runs rescored. Never reword a question to make it pass.
6. Write a dated section in `ROADMAP.md` (what was measured, the numbers with intervals, what can and cannot be claimed) and comment the key numbers on #167.

### Step 4: experiments, on `tune` only

For each candidate, in this order of value: the retrieval candidates of #146 (the question translated into Danish as a second query, smaller units in definition lists, re-ranking), the language-versus-structure comparisons of #168 (the five Danish/English pairs in `tune`, retrieval-only), the scope check of #150 (the three `tune` refusals must be refused; none of the 50 answerable `tune` questions may be refused). The borderline questions that sound out of scope but are not (aircrew ×3, building materials) all share chunks and landed in `confirm`, so the scope check meets them only at confirmation; if it needs tuning against that trap, write new borderline questions for `tune` instead of moving these.

1. **Commit the adoption rule before measuring** (ROADMAP or the issue). Template: retrieval-only first, the reciprocal-rank sign test against the retrieval baseline must be significant on `bge-multilingual-gemma2` (the dashboard computes it) and evidence recall must not drop; then a full `tune` run where the pass/fail sign test against the `tune` baseline is significant, net improvements exceed the flips between the two baselines, and no refusal fault is new; then the Mistral rescore, under which the same rule must hold.
2. Run the `bge-m3` retrieval-only counter-check and report it next to the decision.
3. A change that meets its rule is switched on by default in its own small PR (against `staging`, only when the maintainer asks).

What 53 questions can show: on pass/fail, only effects of about 6 or more consistent net flips (8 of 9, 9 of 10); smaller real effects show in retrieval ranks, not in the pass rate. A clear "does not help" is a valid result.

### Step 5: confirmation and frontier comparison (ask first)

1. **Confirm the optimised configuration on `confirm`**, once, with the rule written before the run: adopt if net flips against the `confirm` baseline are positive and there is no significant regression. At most three confirmations per set of confirm questions.
2. **Gemini 3.8 Flash on `confirm`**, with the same final pipeline. Code needed first (TDD): add `gemini-3.8-flash` to `_GEMINI_MODELS` in `graph/llm_factory.py`; for Gemini 3 and later use temperature 1.0 (Google's guidance; lower values can loop) and pin the thinking level to the documented default `medium`, recorded in the run config. Smoke-test it with `--limit 2 --no-history` and check the cost before the full run. Gemini 3.8 Flash costs $0.75 / $3.75 per million tokens through 2026-12-31, then double.
3. Rescore both `confirm` runs with Mistral. Report the gap with intervals and per error type: if Flash fails the same retrieval misses, the bottleneck is retrieval, not the open model.

## 7. Rules

- Do not tune on `confirm`, and do not read its failures question by question; that turns it into development data (eval/README.md).
- Never move an item between splits after a run; never delete items. New questions go in without a split and `uv run python -m eval.split` places them; new questions change the `questions_hash`, so baselines are run again.
- Do not reword questions to make them pass; a genuinely ambiguous question is a finding: report it, fix it, rescore.
- After any re-ingestion or new documents (#169): run `--check-index`; record split leaks, do not fix them by moving items; check the six refusals again (new documents can answer them).
- Conventional Commits; never edit version numbers. Pull requests against `staging`, only when the maintainer asks.

## 8. Known soft spots (check before blaming the system)

- `test-dk-lukket-kilde-anbefalet-anvendelsesperiode`: vital nugget 1 is an absence ("no fixed maximum period"); the judge might flag a correct "no maximum" as unsupported.
- Refusals with related information in the corpus (`test-refuse-skift-p-nummer`, `test-refuse-sst-id`): a hedged answer that names the general rule and says the specific point is not given counts as a refusal under the judge's definition; if the judge disagrees, record it as a judge issue.
- Questions on table chunks: the evidence is findable; whether the model reads "row, column = value" chunks correctly is one of the things to learn.
- Difficulty ratings come from reviewing models, not from systems.

## 9. Ask before

- Changing any question, expected answer or nugget importance (evidence fixes on `tune` are fine, recorded).
- Moving or deleting items, or re-running `eval.split` on assigned items.
- Re-ingesting documents.
- Step 5, and any spending beyond the budget in section 4.
- Opening a pull request, or merging.
- Cleaning the five dev questions in `golden.json` that still point to the source (it would make earlier runs incomparable).

## 10. Related

#167 (held-out questions, this work), #146 (retrieval candidates), #168 (language versus structure), #150 (scope check), #133 (labelling), #169 (SST documents for the corpus), #170 (chat report export, separate).
