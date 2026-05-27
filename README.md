# Radiation Safety RAG

![Alpha](https://img.shields.io/badge/status-alpha-orange)
[![CI](https://github.com/eikrad/Radiationsafety/actions/workflows/ci.yml/badge.svg)](https://github.com/eikrad/Radiationsafety/actions/workflows/ci.yml)

RAG system for querying IAEA and Danish radiation safety documents. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Architecture

The architecture includes an API pre-processing stage and a [LangGraph](https://langchain-ai.github.io/langgraph/) execution stage. The API validates inputs, short-circuits non-question acknowledgements, resolves provider/API-key settings, and then invokes the graph for retrieval, document grading, optional extra retrieval and web-search fallback, generation, grounding retries, and trusted-source verification.

![RAG flow](architecture.svg)

Diagram source: `architecture.mmd` (Mermaid). To regenerate: `uv run python scripts/render_architecture.py`.

## Running with Docker

The image does not ship the vector DB (`.chroma` is too large for the repo). Run ingestion once, then use the app.

1. Copy `.env.example` to `.env`. Set **`GOOGLE_API_KEY`** (required for ingestion and retrieval). Optionally set `LLM_PROVIDER` and the matching key for generation (`GOOGLE_API_KEY`, `MISTRAL_API_KEY`, or `OPENAI_API_KEY`).
2. From the project root:
   ```bash
   docker compose up --build
   ```
3. In another terminal, run ingestion once (fills the persisted `chroma_data` volume):
   ```bash
   docker compose run --rm backend python ingestion.py
   ```
   Wait for it to finish, then stop the stack (`Ctrl+C`) and start again with `docker compose up` so the backend loads the new DB.
4. Open **http://localhost:8080** for the UI. The frontend proxies `/api` to the backend.

The backend uses a named volume `chroma_data` for `.chroma`, so you only need to run ingestion once per environment. To refresh the document base (e.g. after adding new sources), run the ingestion command again. Changing `LLM_PROVIDER` does not require re-running ingestion.

## Setup

1. Copy `.env.example` to `.env` and configure:
   - **Embeddings (for ingestion and retrieval):** The vector store uses **Gemini embeddings** only. You need **`GOOGLE_API_KEY`** to run ingestion and to run queries (retrieval). Set this even if you later choose OpenAI or Mistral as the LLM for answers.
   - **LLM for generation:** `LLM_PROVIDER` = `gemini`, `mistral`, or `openai`. Set the matching key: `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, or `OPENAI_API_KEY`. The same vector store (Gemini embeddings) is used for retrieval regardless of which LLM you use for generation.
   - Optional: `WEB_SEARCH_ENABLED=true`, `BRAVE_SEARCH_API_KEY` for fallback; `WEB_SEARCH_TRUSTED_DOMAINS_ONLY=true` to restrict web search to iaea.org/retsinformation.dk/sst.dk (default is unrestricted; answers are still verified against trusted sources)
   - Optional: `LANGCHAIN_API_KEY` for LangSmith tracing (tracing is auto-disabled when API keys are sent from the frontend to avoid leaking keys to LangSmith)

2. Install dependencies:
   ```bash
   uv sync
   ```

3. **(Optional)** Document sources: copy `document_sources.example.yaml` to `document_sources.yaml` and add URLs, or **build the list from local PDFs** (see "Building document_sources.yaml from local PDFs" below). `document_sources.yaml` is gitignored by default so you can keep local or repo-specific URLs; remove that line from `.gitignore` if you want to commit a shared registry. The **Documents** button in the UI checks for updates (e.g. retsinformation.dk "Senere ændringer", IAEA "Superseded by") When you re-run ingestion, Danish sources always use the **newest** version of the series; the registry file is updated with that URL. Older Danish versions are kept in `documents/backup/Bekendtgørelse` (at most 2 per source).

4. Run ingestion (requires **`GOOGLE_API_KEY`**; embeddings are always Gemini):
   ```bash
   uv run python ingestion.py
   ```
   You only need to run ingestion once per document set. Changing `LLM_PROVIDER` (e.g. to OpenAI or Mistral for generation) does **not** require re-running ingestion—the same vector store is used.
   Ingestion loads **(1) local PDFs** from `documents/IAEA`, `documents/IAEA_other`, `documents/Bekendtgørelse`, and **(2) documents from URLs** listed in `document_sources.yaml`: **Danish** sources are fetched as **XML** from retsinformation.dk (newest version of the series), IAEA sources from the publication page PDF link, and any direct PDF URLs. You can rely entirely on the registry and skip placing PDFs locally. Use the **Documents** panel in the UI to "Check for updates" and "Re-run ingestion".

5. Start backend:
   ```bash
   uv run uvicorn api.main:app --reload --port 8000
   ```

6. Frontend – from project root, choose one:
   - **Single server**: `npm -C frontend run build` then open http://localhost:8000
   - **Dev mode** (hot reload): `npm -C frontend install && npm -C frontend run dev` then open http://localhost:5173

7. Optional CLI:
   ```bash
   uv run python main.py
   ```

## Evaluation

The evaluation harness lives in **`eval/`**. It runs the RAG graph on a golden Q&A dataset and scores outputs with RAGAS-style metrics (faithfulness, answer relevance, context precision, context recall), writing markdown and JSON reports to `eval/reports/`.

From the project root:

```bash
uv run python -m eval.run_eval
```

The harness uses your `.env` for the LLM (no API keys in the golden data). Run ingestion first so the graph has documents to retrieve. See `eval/README.md` for options (`--limit`, `--no-web-search`), metric definitions, and optional LangSmith tracing.

## Testing

CI runs the test suite on push and on pull requests (see status badge above).

- **Backend**: `uv pip install -e ".[dev]"` then `uv run pytest tests/ -v`
- **Frontend**: `cd frontend && npm run test` (or `npm run test:watch` for watch mode)

## Security and Operations

- Mutating routes (`/ingest` and mutating `/documents/*` endpoints) require `X-Admin-Token`.
- If `ADMIN_TOKEN` is not configured, admin routes are fail-closed (`503`) unless `ADMIN_AUTH_BYPASS=true` is explicitly set for local-only use.
- Query/admin rate limits are in-memory and per-client (`RATE_LIMIT_*`). This MVP is suitable for single-process deployments.
- For multi-worker or multi-replica deployments, set `RATE_LIMIT_BACKEND=redis` and `RATE_LIMIT_REDIS_URL` to enforce global limits.

### Runbook quick checks

- **429 spike**: verify `RATE_LIMIT_*` values and recent traffic source patterns.
- **Admin routes suddenly unavailable (503)**: check `ADMIN_TOKEN` presence and accidental bypass misconfiguration.
- **Service health degraded**: inspect `/health` and `/metrics` plus container health status in Compose.

### Container hardening notes

- Backend container runs as non-root user (`appuser`) and sets `PYTHONDONTWRITEBYTECODE=1` plus `PYTHONUNBUFFERED=1`.
- Compose applies `no-new-privileges` and drops Linux capabilities (`cap_drop: [ALL]`) for backend/frontend.
- Backend uses `tmpfs: /tmp` and a persistent named volume only for `/app/.chroma`.
- Healthchecks are active for backend, and frontend waits for backend healthy state before startup.

## Building document_sources.yaml from local PDFs

To populate `document_sources.yaml` from the PDFs you already have in `documents/`:

```bash
uv run python build_document_sources.py
```

This scans `documents/IAEA`, `documents/IAEA_other`, and `documents/Bekendtgørelse`, extracts titles and version info from PDF metadata and first-page text (and from Danish `*_version.txt` files), optionally confirms Danish ELI URLs on retsinformation.dk, merges with any existing registry entries (to keep URLs), and writes the full list to `document_sources.yaml`. Use `--no-confirm` to skip URL lookups, or `--dry-run` to print the list without writing.

## Collections and embeddings

- **`radiation-iaea`**: IAEA and IAEA_other PDFs  
- **`radiation-dk-law`**: Bekendtgørelse (Danish legislation), ingested from retsinformation.dk XML (newest version)

Retrieval always uses **Gemini embeddings** (one shared vector store). The LLM that generates answers can be Gemini, OpenAI, or Mistral. The LLM only receives the **retrieved text** (the chunks found by similarity search); it never sees or interprets the embedding vectors. So OpenAI (or Mistral) can be used for generation while the store stays on Gemini embeddings—no re-ingestion needed. Embedding models from different providers use different dimensions and are not interchangeable; adding OpenAI as an optional *embedding* backend would require separate Chroma collections and could be added later if needed.

## Dependency notes

**2026-05-27 — Weekly maintenance**

### Fixes applied

- **`black` and `isort` moved to dev-only** — they were incorrectly listed as runtime dependencies. They are code-formatting tools and belong in `[project.optional-dependencies] dev`. Production installs (`uv sync` without `--all-extras`) are now leaner.
- **CI**: `actions/checkout` updated from `v5` to `v6` (current stable).

### Major upgrades available (not auto-applied — require testing)

These packages have new major versions that were not auto-applied because major bumps may contain breaking API or config changes:

| Package | In use | Latest | Notes |
|---|---|---|---|
| `vite` (frontend) | `^7.3.1` | `8.x` | New config/plugin APIs; review migration guide |
| `@vitejs/plugin-react` | `^5.1.1` | `6.x` | Follows Vite major |
| `eslint` / `@eslint/js` | `^9.x` | `10.x` | Flat-config updates |
| `typescript` | `~5.6` | `6.x` | New type-system features; some breaking changes |
| `langchain-google-genai` | `>=2.0.0` | `4.x` | Two major versions ahead — review the LangChain changelog before upgrading |

## Credits and references

This project was inspired by and draws on patterns from the **LangChain / LangGraph course** by **Eden Marco** and the accompanying open-source repository:

- **Eden Marco** – [LangChain course](https://github.com/emarco177/langchain-course) (GitHub)
- Repository: [github.com/emarco177/langchain-course](https://github.com/emarco177/langchain-course) (Apache-2.0)

We thank [Roman Kuznetsov (@kuznero)](https://github.com/kuznero) for valuable comments on the project.
