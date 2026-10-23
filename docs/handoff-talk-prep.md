# Handoff: open measurements before the talk on 2026-10-30

For a new session. Read this file first; it stands alone. Written 2026-10-23. It lists what is still to be **tested**, not what goes on slides.

**Paste this to start the session:**

> Read `docs/handoff-talk-prep.md` in this repository and carry out the tasks marked "agent". Stop and ask me before anything under "Ask before", and for every task marked "maintainer" give me the exact commands instead of running them.

## 1. Why these tests

The results are shown in a 15-minute talk on 2026-10-30. Three claims need data that does not exist yet or contradict what the data say:

- "Small local models lag behind larger ones": the only local numbers are from the dev set on 2026-10-11, before the correction loop and with another setup.
- "Retrieval problems mainly in the Danish legal texts": true on the old dev set; on the held-out `tune` questions IAEA questions fail more often (9 of 16 pass) than Danish-law ones (25 of 33), and Danish law asked in English passes better (6 of 7) than in Danish (19 of 26). Whether structure (tables, cross-references, multi-part questions) rather than language is the cause is untested (#168).
- The correction loop gained 3 questions on `tune` (37 → 40 under both judges) but was not confirmed on `confirm`; one open question (Mistral finds more unsupported claims after it) is unexplained.

## 2. Current state (all on `origin/feature/confirm-and-frontier`, which contains `staging`; not yet merged)

Configuration unless stated: `gemma-4-26b-a4b-it`, `bge-multilingual-gemma2`, k = 5, no web search, judge `qwen3.8-27b` with 3 votes, second judge `mistral-medium-3.5-128b`. Source: `ROADMAP.md` and `eval/history/runs.jsonl`.

| What | Result | Run IDs |
|---|---|---|
| Dev set, no web | 36/39 = 92 % | `20261016_153423` |
| `tune` baseline, two runs | 70 % / 72 % (Qwen), 68 % (Mistral) | `20261016_182025`, `_183512`, `_184941` |
| `confirm` baseline | 42/53 = 79 % (Qwen), 45/53 (Mistral) | `20261016_173446`, `20261017_203957` |
| Loop on `tune` | 37 → 40 (Qwen), 36 → 40 (Mistral); retrieval misses 12 → 7 | `20261017_143922`, `_154702` |
| Loop on `confirm` | not confirmed: Qwen 3 better / 1 worse, Mistral 3 / 4 | `20261017_191022`, `_203622` |
| Gemini 3.8 Flash on `confirm`, same pipeline | 89 % (Qwen), 85 % (Mistral); intervals overlap Gemma's 83 %; same 2 of 3 retrieval misses | `20261017_190626`, `_203329` |
| Failed `tune` questions | 12 retrieval misses (10 with evidence at rank 6–20), 2 generation faults, 2 judge noise, 1 label problem | ROADMAP |
| Re-ranking 20 candidates (#146) | rule not met (13 better / 5 worse, p = 0.10) | branch `feature/rerank-candidates` |

## 3. Rules

- `confirm` is read **in aggregate only** (pass rates with intervals, counts per error type, number of flips), never question by question and never to tune. At most three confirmations of the optimised pipeline; **one is used**. A comparison of another model on `confirm` is "reported, not decided on"; write that rule into the ROADMAP **before** the run.
- Every experiment has its adoption rule committed before it is measured (ROADMAP or the issue). A clear "does not help" is a valid result.
- Experiments run on `tune` only; retrieval-only first where that can decide (cheap), a full run only for what passes.
- Do not read or print `.env`. The dashboard shows the cost of each run (`eval/data/prices.json`); stop and ask if a run costs clearly more than its estimate.
- Commit before every recorded run (a dirty tree is refused); commit `eval/history/runs.jsonl` and the new archive in `eval/history/outputs/` together. Conventional Commits. PRs target `staging`, opened only when the maintainer asks.
- Report numbers with 95 % intervals. One question is about 2 points on 53; do not rank results whose intervals overlap. A pass/fail effect under about 3 net questions is within noise.

## 4. Tests

### T1. Local model on `confirm`, same protocol (maintainer; the agent prepares the rule and the commands)

`qwen3.5:4b` with `bge-m3`, the same pipeline as the Gemma and Gemini runs including the correction loop (it works without web search), on `confirm`, reported only. The agent first adds to `ROADMAP.md`: "reported, not decided on; not a confirmation of the optimised pipeline; `confirm` stays aggregate only". The maintainer then runs it on the machine with Ollama (variables: see `.env.example` and the privacy-mode section of `README.md`; context window about 12288 as in the earlier local runs):

```bash
export EVAL_GRADER_PROVIDER=scaleway EVAL_JUDGE_MODEL=qwen3.8-27b EVAL_JUDGE_VOTES=3
# LLM_PROVIDER=ollama, OLLAMA_MODEL=qwen3.5:4b, bge-m3 collections as in privacy mode
uv run python -m eval.run_eval --golden eval/data/golden_test.json --split confirm \
  --no-web-search --label confirm-local-qwen35-4b --notes "local model, reported only"
EVAL_JUDGE_MODEL=mistral-medium-3.5-128b uv run python -m eval.run_eval \
  --golden eval/data/golden_test.json --rescore <RUN_ID> --label confirm-local-qwen35-4b-judge-mistral
```

Allow one to two hours (the dev-set run without the loop took 38 minutes). Record time per question and the share of the model on the GPU. Compare with Gemma (83 %) and Flash (89 % / 85 %) with intervals, and by error type: does the local model fail on the same retrieval misses, or on generation?

### T2. Why Mistral finds more unsupported claims after the loop (agent, no API keys, `tune` only)

Compare `tune-local-loop` (`20261017_143922`, Qwen) and its Mistral rescore (`20261017_154702`) with `tune-baseline` (`20261016_160821` / rescore `20261016_182025`) and its Mistral rescore (`20261016_184941`), using the archived outputs in `eval/history/outputs/`. Find which questions changed verdict between the judges, whether they took the loop path (`missing_queries`, node path), and whether the extra claims come from follow-up chunks in the context or from the answer. Write the finding into a dated section of `ROADMAP.md`; say plainly if it cannot be explained.

### T3. Language or structure (#168, maintainer for runs, agent for the analysis)

Retrieval-only on `tune`, evidence ranks per question, one factor at a time; adoption rule as in #168 (sign test on ranks, otherwise "not separable at this set size"):
1. From existing data (agent, free): break the 53 `tune` and the 39 dev questions down by source, question language and question type (table, multi-part, cross-reference, definition, single paragraph) against the retrieval result; report which factor separates misses from hits. `confirm` is not used.
2. The five Danish/English question pairs in `tune`, retrieval-only with both embedders (`bge-multilingual-gemma2`, `bge-m3`).
3. PDF copy versus XML copy of the Danish orders in a separate collection (needs an index build; ask first).
Step 3 is optional for the talk; 1 and 2 are cheap and enough to answer "language or structure?" with a hint.

### T4. Optional: one retrieval-only candidate (maintainer decides; skip if not started by 2026-10-26)

From "Next round on `tune`" in the ROADMAP: re-rank 50 candidates (`RERANKER=bge-reranker-v2-m3`, `RERANK_CANDIDATES=50`), neighbour chunks, or BM25 + dense fusion for exact terms. Commit the rule first (same as the first re-ranking round), run retrieval-only on `tune` against `tune-retrieval-bge-gemma2`, report the sign test on ranks and the effect on evidence recall in context. No full runs and nothing on `confirm`.

## 5. Ask before

- Merging or opening a PR for `feature/confirm-and-frontier`.
- Any run that costs money, or that uses a `confirm` confirmation.
- Re-ingestion or any change to the index (fingerprints `b0fc52f61a4e` / `d90d785f314c` must stay).

## 6. Done when

- T1 is run and recorded (or dropped with the reason, and the dev-set local numbers are then labelled as an earlier setup), T2 and T3.1–3.2 are written up in `ROADMAP.md` with the numbers and intervals, T4 is either measured under its committed rule or not started. The maintainer has the results by 2026-10-27.
