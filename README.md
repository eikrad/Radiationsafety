# Radiation Safety RAG

![Alpha](https://img.shields.io/badge/status-alpha-orange)

RAG system for querying IAEA and Danish radiation safety documents.

## Setup

1. Copy `.env.example` to `.env` and configure:
   - `LLM_PROVIDER`: `gemini` or `mistral`
   - `GOOGLE_API_KEY` or `MISTRAL_API_KEY` (depending on provider)
   - Optional: `WEB_SEARCH_ENABLED=true`, `BRAVE_SEARCH_API_KEY` for fallback
   - Optional: `LANGCHAIN_API_KEY` for LangSmith tracing

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

5. Start frontend:
   ```bash
   cd frontend && npm install && npm run dev
   ```

6. Optional CLI:
   ```bash
   uv run python main.py
   ```

## Collections

- `radiation-iaea`: IAEA and IAEA_other PDFs
- `radiation-dk-law`: Bekendtgørelse (Danish legislation) PDFs
