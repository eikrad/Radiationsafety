# Evaluation harness

Systematic evaluation for the radiation safety RAG: golden Q&A dataset, RAGAS-style metrics, and report generation.

## How to run

From the project root:

```bash
uv run python -m eval.run_eval
```

The harness uses your `.env` for the LLM (no API keys in golden data). Ensure ingestion has been run so the graph has documents to retrieve.

### Options

| Option | Description |
|--------|-------------|
| `--golden PATH` | Path to golden JSON (default: `eval/data/golden.json`) |
| `--limit N` | Run only on the first N items (useful for development) |
| `--no-web-search` | Disable web search for reproducible eval runs |
| `--output-dir PATH` | Directory for report files (default: `eval/reports`) |
| `--cache-dir PATH` | Cache graph outputs here (or set `EVAL_CACHE_DIR`); re-runs skip graph and only recompute metrics when cache is valid |
| `--per-chunk-precision` | Use per-chunk context precision (one LLM call per question, precision@1/3/5) instead of single sufficiency call |
| `--pass-rule all\|mean` | Pass when **all** metrics ≥ 0.5 (`all`, default) or when **mean** of metrics ≥ 0.5 (`mean`). Use `mean` if one strict metric (e.g. context precision) dominates. |
| `--delay-after-graph SEC` | Seconds to wait after each graph run before running metrics (default 5). Overrides `EVAL_DELAY_AFTER_GRAPH_SEC`. Use `0` to disable. |
| `--delay-between-items SEC` | Seconds to wait between processing each golden item (default 20). Overrides `EVAL_DELAY_BETWEEN_ITEMS_SEC`. Use `0` to disable. |

**Rate limits:** By default the runner waits **5 s** after each graph invoke (before the 4 metric LLM calls) and **20 s** between items so eval stays under typical free-tier limits (e.g. Mistral ~1 RPS, ~30 RPM). Set the env vars above or use `--delay-after-graph 0 --delay-between-items 0` to disable delays.

Example:

```bash
uv run python -m eval.run_eval --limit 3 --no-web-search
```

## Environment

- **Retrieval**: The graph uses **Gemini embeddings** for retrieval; set **`GOOGLE_API_KEY`** in `.env`. Run ingestion once so the vector store is populated.
- **Generation**: The graph uses `LLM_PROVIDER` and the corresponding API key for answer generation (e.g. `mistral` + `MISTRAL_API_KEY`, or `openai` + `OPENAI_API_KEY`, or `gemini` + `GOOGLE_API_KEY`).
- **Grading LLM** (optional): Set `EVAL_GRADER_PROVIDER=gemini` (or `openai` / `mistral`) to use a different model only for computing metrics (faithfulness, answer relevance, context precision, context recall). Example: use Mistral for answering and Gemini for grading by setting `LLM_PROVIDER=mistral` and `EVAL_GRADER_PROVIDER=gemini`; ensure `MISTRAL_API_KEY` and `GOOGLE_API_KEY` are in `.env` (Gemini is required for retrieval in any case).
- **Eval delays** (optional): `EVAL_DELAY_AFTER_GRAPH_SEC` and `EVAL_DELAY_BETWEEN_ITEMS_SEC` override the default 5 s and 20 s delays used to avoid LLM rate limits. Set to `0` to disable.
- **Optional – LangSmith**: If `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set, eval runs are traced; use tags `eval` and `golden` in the LangSmith UI to filter. See [LangSmith](#langsmith) below.

## Metrics

Each run computes four metrics (0–1 per question, then averaged):

| Metric | Meaning |
|--------|--------|
| **Faithfulness** | The answer is grounded in the **retrieved context** (no hallucination). An LLM grader compares the generation to the same documents the generator saw and scores whether the factual content is supported. |
| **Answer relevance** | The answer addresses the question. |
| **Context precision** | The retrieved context is sufficient to answer the question (top-k sufficiency). |
| **Context recall** | Key facts from the golden set appear in the retrieved context (or sufficiency as proxy if no key facts). |

**Faithfulness vs. the graph’s “trusted” check:** The graph node `verify_trusted` checks the answer only against **trusted** sources (vector DB: IAEA + Danish law) and can set `retrieval_warning` or trigger web search. The **eval faithfulness** metric is different: it only checks “is the answer supported by whatever context was given to the generator?” (RAGAS-style). It does not care whether that context was from the vector DB or from web search.

By default a question **passes** only if all four metrics are ≥ 0.5. With `--pass-rule mean`, it passes when the mean of the four metrics is ≥ 0.5 (useful when one metric, e.g. context precision, is often strict). The report shows pass rate and per-metric means, plus per-question details.

## Output

- **JSON**: `eval/reports/report_<timestamp>.json` – machine-readable summary and per-question results.
- **Markdown**: `eval/reports/report_<timestamp>.md` – human-readable summary and per-question breakdown.


## LangSmith

To trace eval runs in LangSmith:

1. Set in `.env`: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=radiation-safety-rag` (or your project), and `LANGCHAIN_API_KEY=...` (from [LangSmith](https://smith.langchain.com)).
2. Run: `uv run python -m eval.run_eval`.
3. In the LangSmith UI, filter runs by tags **eval** and **golden** to see evaluation runs. Each graph invoke and metric call is traced so you can inspect retrieval, generation, and grading per question.
