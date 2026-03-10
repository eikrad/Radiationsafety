# Evaluation harness

Systematic evaluation for the radiation safety RAG: golden Q&A dataset, RAGAS-style metrics, and report generation.

## How to run

From the project root:

```bash
uv run python -m eval.run_eval
```

Uses `.env` for the LLM (no API keys in golden data). See main project README for setup.

## Options

- `--limit N` – run on first N golden items (for development).
- `--no-web-search` – disable web search for reproducible eval (or set `WEB_SEARCH_ENABLED=false`).
- Golden data: `eval/data/golden.json`.

More details (env vars, metrics, LangSmith) will be added as the harness is completed.
