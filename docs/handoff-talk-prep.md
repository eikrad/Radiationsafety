# Handoff: data and figures for the Scientific Friday Bar talk (2026-10-30)

For a new session. Read this file first; it stands alone. Written 2026-10-23.

**Paste this to start the session:**

> Read `docs/handoff-talk-prep.md` in this repository and carry out the tasks marked "agent". Stop and ask me before anything under "Ask before", and for every task marked "maintainer" give me the exact commands instead of running them.

## 1. Context

- **Event:** Scientific Friday Bar #2 (scifbar.dk), Friday 2026-10-30, GEUS, Copenhagen, about 20 people, English, data scientists and domain researchers. The talk is **15 minutes** with a hard stop, then open discussion. The audience wants real use cases with measurements and wants to challenge them.
- **Abstract (already on the website, third person, no numbers):** a RAG system for radiation safety (international safety standards and Danish law) evaluated step by step with a golden set and an independent LLM judge; retrieval problems mainly in the Danish legal texts; small local models lagging behind larger ones; whether the Danish problems stem from language or from document structure still open; transferable to other domains with complex structured documents (geology, risk management, compliance).
- **The abstract's claims versus the data since 2026-10-16:** "mainly in the Danish legal texts" holds on the old development set only. On the held-out `tune` questions IAEA questions fail more often (9 of 16 pass) than Danish-law ones (25 of 33), and Danish law asked in English passes better (6 of 7) than asked in Danish (19 of 26). The talk should say so openly: it answers the "language or structure" question with "the data point to tables, multi-part questions and cross-references, not the language".
- **Thesis for the talk (proposal):** "We tuned a RAG to 97 % on our own questions; on fresh questions it passed about 75 %. What we measured after that was not what we expected." Everything below serves that story.

## 2. What is measured (all on `origin/feature/confirm-and-frontier`, which contains `staging`; not yet merged)

Configuration unless stated: `gemma-4-26b-a4b-it`, `bge-multilingual-gemma2`, k = 5, no web search, judge `qwen3.8-27b` with 3 votes, second judge `mistral-medium-3.5-128b`. Source of every number: `ROADMAP.md` (sections "Held-out test set — baselines", "The local correction loop never ran", "Held-out step 5") and `eval/history/runs.jsonl`.

| What | Result | Run IDs / where |
|---|---|---|
| Dev set (39 questions), no web | 36/39 = 92 % (80–97 %); with web `prompt-145` 38/39 | `20261016_153423` |
| Held-out `tune` baseline, two runs | 70 % and 72 % (Qwen), 68 % (Mistral) | `20261016_182025`, `_183512`, `_184941` |
| Held-out `confirm` baseline | 42/53 = 79 % (Qwen), 45/53 (Mistral) | `20261016_173446`, `20261017_203957` |
| Test overall (`tune` + `confirm`) | 79/106 = 75 % (65–82 %); 76 % without the 14 items on dev chunks | ROADMAP |
| Noise floor | 1 question flips between identical `tune` runs; effects under about 3 net questions are not distinguishable | ROADMAP |
| Correction loop never ran (web-search gating) | after the fix on `tune`: 37 → 40 (Qwen), 36 → 40 (Mistral), retrieval misses 12 → 7, cost €1.08 → €1.40 | `20261017_143922`, `_154702` |
| Loop on `confirm` | **not confirmed**: Qwen 3 better / 1 worse (p = 0.63), Mistral 3 / 4 (p = 1.0) | `20261017_191022`, `_203622` |
| Gemini 3.8 Flash on `confirm`, same pipeline | 47/53 = 89 % (Qwen), 45/53 = 85 % (Mistral) vs Gemma 83 % / 83 %; all intervals overlap; 2 of 3 retrieval misses are the same questions; answers cost $4.40 vs €0.72 | `20261017_190626`, `_203329` |
| Failed `tune` questions, classified | 12 retrieval misses (10 with the evidence at rank 6–20, recall@20 0.97, recall@5 0.88), 2 generation faults, 2 judge noise, 1 label problem | ROADMAP |
| Re-ranking 20 candidates (#146) | rule not met (13 better / 5 worse, p = 0.10) | branch `feature/rerank-candidates` |
| Local answer models on the **dev** set, 2026-10-11 (before the loop, other setup) | `llama3.1:8b` 20–22/39, `qwen3.5:4b` 22–23/39 vs 38/39 cloud | ROADMAP, privacy mode |

## 3. Rules that must hold

- `confirm` is read **in aggregate only** (pass rates with intervals, counts per error type, number of flips). Never question by question, never to tune; that turns it into development data (`eval/README.md`). At most three confirmations of the optimised pipeline; **one is used**. A comparison of other models on `confirm` is "reported, not decided on", like the Gemini run; write that rule into the ROADMAP **before** the run.
- Questions shown in the talk as examples come from `tune` or the dev set, never from `confirm`.
- Do not read or print `.env`. Costs are shown by the dashboard (`eval/data/prices.json`); stop and ask if a run costs clearly more than its estimate.
- Commit before every recorded run (a dirty tree is refused); commit `eval/history/runs.jsonl` and the new archive in `eval/history/outputs/` together. Conventional Commits. PRs target `staging`, opened only when the maintainer asks.
- Report numbers with their 95 % intervals. One question is about 2 points on 53 questions; do not rank models whose intervals overlap.

## 4. Tasks

### A. Why Mistral finds more unsupported claims after the loop (agent, no API keys, `tune` only)

Open question from step 5: under Mistral `unsupported_claim` rises 1 → 5 after the loop on `confirm`, under Qwen it does not. Investigate on **`tune`**, never on `confirm`: `tune-local-loop` (`20261017_143922`, Qwen) and its Mistral rescore (`20261017_154702`) against `tune-baseline` (`20261016_160821` / rescore `20261016_182025`) and its Mistral rescore (`20261016_184941`). Use the archived outputs in `eval/history/outputs/`. Find which questions changed verdict between the judges, whether they took the loop path (`missing_queries`, node path), and whether the extra claims come from the follow-up chunks in the context or from the answer. Write the finding (a few paragraphs, with question IDs) into a dated section of `ROADMAP.md`; say plainly if it cannot be explained.

### B. Figures for the slides (agent, from the committed history)

Load the `dataviz` skill first. Four figures as PNG and SVG in `docs/talk/figures/`, each with a source line naming the run IDs:
1. Pass rate with 95 % Wilson intervals: dev (no web), `tune` baseline, `confirm` baseline, test overall.
2. Models on `confirm` (Gemma, Gemini 3.8 Flash, and the local model once task C exists): pass rate with intervals under both judges, and the answer cost per 53 questions. Mark that the intervals overlap.
3. The 17 classified `tune` failures by class (retrieval miss, generation fault, judge noise, label problem), with the retrieval misses split into evidence ranked 6–20 versus not in the top 20.
4. Evidence recall by rank (recall@1/3/5/10/20) for the dev set (0.53 / 0.88 / 0.92 / 0.94 / 0.97, run `20260930_113512`) and for `tune` (recall@5 0.88, @20 0.97; runs `20261016_185213`).

Colour-blind safe, readable at the back of a room, no decoration. Use only aggregate numbers for `confirm`.

### C. Local model with the same protocol (maintainer; the agent prepares the rule and the commands)

The abstract promises "small local models lag behind larger ones", and the only local numbers are from the dev set before the loop. Run `qwen3.5:4b` with `bge-m3` on `confirm` with the same pipeline (including the correction loop, which works without web search), reported only. Before the run the agent adds to `ROADMAP.md`: "reported, not decided on; it does not count as a confirmation of the optimised pipeline; `confirm` stays aggregate only". The maintainer then runs, on the machine with Ollama (see `.env.example` and the privacy-mode section of `README.md` for the variables; about 12288-token window as in the earlier local runs):

```bash
export EVAL_GRADER_PROVIDER=scaleway EVAL_JUDGE_MODEL=qwen3.8-27b EVAL_JUDGE_VOTES=3
# LLM_PROVIDER=ollama, OLLAMA_MODEL=qwen3.5:4b, bge-m3 collections as in privacy mode
uv run python -m eval.run_eval --golden eval/data/golden_test.json --split confirm \
  --no-web-search --label confirm-local-qwen35-4b --notes "local model, reported only"
EVAL_JUDGE_MODEL=mistral-medium-3.5-128b uv run python -m eval.run_eval \
  --golden eval/data/golden_test.json --rescore <RUN_ID> --label confirm-local-qwen35-4b-judge-mistral
```

Allow one to two hours (the dev-set run without the loop took 38 minutes). Report time per question and the share of the model on the GPU next to the pass rate. If it cannot finish before 2026-10-27, show the dev-set numbers and say they come from an earlier setup.

### D. One concrete loop example for the slides (agent, from archives)

Pick a `tune` question where the first retrieval is insufficient and the loop then finds the evidence, for example `test-dk-cs137-sikringsgruppe-c-krav` (Mistral passes it after three follow-ups, the first "Bilag 6 Bekendtgørelse om radioaktive stoffer sikringsgrupper"; check the exact ID in `golden_test.json`). Extract from the archive: the question, the first retrieval's gap (`needed`/`found`), each follow-up query, and the answers before and after. Write it as a half-page `docs/talk/loop-example.md`. The maintainer can add a screen recording as a fallback in case a live demo fails.

### E. Slide outline for 15 minutes (agent)

One-page `docs/talk/outline.md`, one line per slide with the figure it uses and the sentence to say. Proposal: problem and documents (1 min); how it is evaluated: golden set, nuggets, independent judge (2); the 97 % and what it was tuned on (1); fresh questions: 75 % (2); where it fails: ranking, tables, cross-references, not the language (3); the loop that never ran, and that the gain did not carry over (2); three models, one gap in cost and none clear in quality (2); what is still open and what to challenge (2). Keep the caveats on the slides: dev versus test, overlapping intervals, judge noise.

### F. Optional: one retrieval-only experiment (maintainer decides; skip if not started by 2026-10-26)

From "Next round on `tune`": re-rank 50 candidates (`RERANKER=bge-reranker-v2-m3`, `RERANK_CANDIDATES=50`) or neighbour chunks. Commit the adoption rule first (same rule as the first re-ranking round), run retrieval-only on `tune` against `tune-retrieval-bge-gemma2`, report a clear "does not help" as a result. No full runs and nothing on `confirm`.

## 5. Ask before

- Merging or opening a PR for `feature/confirm-and-frontier` (the maintainer decides).
- Any run that costs money or that uses a `confirm` confirmation.
- Re-ingestion or changing the index (fingerprints `b0fc52f61a4e` / `d90d785f314c` must stay).
- Changing the abstract; it is on the website. Differences between it and the data are handled in the talk.

## 6. Done when

- A and D are written up in `ROADMAP.md` and `docs/talk/`; B's four figures exist with sources; E exists; C is either run and recorded or explicitly dropped with the reason; the maintainer has the slide outline and the figures by 2026-10-27.
