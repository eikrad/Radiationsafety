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

Example:

```bash
uv run python -m eval.run_eval --limit 3 --no-web-search
```

## Environment

- **LLM**: Same as the main app. Set `LLM_PROVIDER` and the corresponding API key (`GOOGLE_API_KEY`, `MISTRAL_API_KEY`, or `OPENAI_API_KEY`) in `.env`.
- **Optional – LangSmith**: If `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set, eval runs are traced; use tags `eval` and `golden` in the LangSmith UI to filter them. See [LangSmith](#langsmith) below.

## Metrics

Each run computes four metrics (0–1 per question, then averaged):

| Metric | Meaning |
|--------|--------|
| **Faithfulness** | The answer is grounded in the retrieved context (no hallucination). |
| **Answer relevance** | The answer addresses the question. |
| **Context precision** | The retrieved context is sufficient to answer the question (top-k sufficiency). |
| **Context recall** | Key facts from the golden set appear in the retrieved context (or sufficiency as proxy if no key facts). |

A question **passes** if all four metrics are ≥ 0.5. The report shows pass rate and per-metric means, plus per-question details.

## Output

- **JSON**: `eval/reports/report_<timestamp>.json` – machine-readable summary and per-question results.
- **Markdown**: `eval/reports/report_<timestamp>.md` – human-readable summary and per-question breakdown.

## LangSmith

To trace eval runs in LangSmith:

1. Set in `.env`: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=radiation-safety-rag` (or your project), and `LANGCHAIN_API_KEY=...`.
2. Run: `uv run python -m eval.run_eval`.
3. In the LangSmith UI, filter runs by tags **eval** and **golden** to see only evaluation runs.

Traces include the full graph invocation per question, so you can inspect retrieval and generation steps.
