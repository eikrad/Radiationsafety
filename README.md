# Radiation Safety RAG

![Alpha](https://img.shields.io/badge/status-alpha-orange)

RAG system for querying IAEA and Danish radiation safety documents.

## Setup

1. Copy `.env.example` to `.env` and configure:
   - `LLM_PROVIDER`: `gemini` or `mistral`
   - `GOOGLE_API_KEY` or `MISTRAL_API_KEY` (depending on provider)
   - Optional: `WEB_SEARCH_ENABLED=true`, `BRAVE_SEARCH_API_KEY` for fallback
   - Optional: `LANGCHAIN_API_KEY` for LangSmith tracing (tracing is auto-disabled when API keys are sent from the frontend to avoid leaking keys to LangSmith)

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run ingestion (requires API key for embeddings):
   ```bash
   uv run python ingestion.py
   ```

4. Start backend:
   ```bash
   uv run uvicorn api.main:app --reload --port 8000
   ```

5. Frontend – choose one:
   - **Single server**: `cd frontend && npm run build` then open http://localhost:8000
   - **Dev mode** (hot reload): `cd frontend && npm install && npm run dev` then open http://localhost:5173

6. Optional CLI:
   ```bash
   uv run python main.py
   ```

## Testing

- **Backend**: `uv pip install -e ".[dev]"` then `uv run pytest tests/ -v`
- **Frontend**: `npm run test` or `npm run test:watch` (from project root), or `cd frontend && npm run test`

## Collections

- `radiation-iaea`: IAEA and IAEA_other PDFs
- `radiation-dk-law`: Bekendtgørelse (Danish legislation) PDFs
