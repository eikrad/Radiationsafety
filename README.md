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

3. **(Optional)** Document sources: copy `document_sources.example.yaml` to `document_sources.yaml` and add URLs, or **build the list from local PDFs** (see “Building document_sources.yaml from local PDFs” below). The **Documents** button in the UI checks for updates (e.g. retsinformation.dk “Senere ændringer”, IAEA “Superseded by”) When you re-run ingestion, Danish sources always use the **newest** version of the series; the registry file is updated with that URL. Older Danish versions are kept in `documents/backup/Bekendtgørelse` (at most 2 per source).

4. Run ingestion (requires API key for embeddings):
   ```bash
   uv run python ingestion.py
   ```
   Ingestion loads **(1) local PDFs** from `documents/IAEA`, `documents/IAEA_other`, `documents/Bekendtgørelse`, and **(2) documents from URLs** listed in `document_sources.yaml`: **Danish** sources are fetched as **XML** from retsinformation.dk (newest version of the series), IAEA sources from the publication page PDF link, and any direct PDF URLs. You can rely entirely on the registry and skip placing PDFs locally. Use the **Documents** panel in the UI to "Check for updates" and “Re-run ingestion”.

5. Start backend:
   ```bash
   uv run uvicorn api.main:app --reload --port 8000
   ```

6. Frontend – choose one:
   - **Single server**: `cd frontend && npm run build` then open http://localhost:8000
   - **Dev mode** (hot reload): `cd frontend && npm install && npm run dev` then open http://localhost:5173

7. Optional CLI:
   ```bash
   uv run python main.py
   ```

## Testing

- **Backend**: `uv pip install -e ".[dev]"` then `uv run pytest tests/ -v`
- **Frontend**: `npm run test` or `npm run test:watch` (from project root), or `cd frontend && npm run test`

## Building document_sources.yaml from local PDFs

To populate `document_sources.yaml` from the PDFs you already have in `documents/`:

```bash
uv run python build_document_sources.py
```

This scans `documents/IAEA`, `documents/IAEA_other`, and `documents/Bekendtgørelse`, extracts titles and version info from PDF metadata and first-page text (and from Danish `*_version.txt` files), optionally confirms Danish ELI URLs on retsinformation.dk, merges with any existing registry entries (to keep URLs), and writes the full list to `document_sources.yaml`. Use `--no-confirm` to skip URL lookups, or `--dry-run` to print the list without writing.

## Collections

- `radiation-iaea`: IAEA and IAEA_other PDFs
- `radiation-dk-law`: Bekendtgørelse (Danish legislation), ingested from retsinformation.dk XML (newest version)

## Credits and references

This project was inspired by and draws on patterns from the **LangChain / LangGraph course** by **Eden Marco** and the accompanying open-source repository:

- **Eden Marco** – [LangChain course](https://github.com/emarco177/langchain-course) (GitHub)
- Repository: [github.com/emarco177/langchain-course](https://github.com/emarco177/langchain-course) (Apache-2.0)
