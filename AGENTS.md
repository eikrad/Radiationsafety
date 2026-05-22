# AGENTS.md — Radiation Safety RAG

Central reference for all AI agents (Claude Code, Codex, Cursor, Gemini CLI, etc.) working on this codebase or querying the Logseq knowledge base.

---

## 1. Project Overview

RAG system for querying IAEA and Danish radiation safety documents.

- **Backend**: FastAPI + LangGraph workflow (`graph/`) + Chroma vector database
- **Embeddings**: always Gemini (`GOOGLE_API_KEY` required for ingestion and retrieval)
- **LLM for generation**: configurable — `gemini`, `openai`, or `mistral` via `LLM_PROVIDER`
- **Frontend**: React/TypeScript in `frontend/`
- **Documents**: `documents/IAEA/`, `documents/IAEA_other/`, `documents/Bekendtgørelse/`

---

## 2. Codebase Map

```
api/main.py              — FastAPI routes, admin auth, rate limiting
graph/graph.py           — LangGraph workflow (nodes, edges, routing)
graph/nodes/             — retrieve, grade_documents, generate, web_search, verify_trusted
graph/chains/            — LLM chains (generation, grading, search-query, truncate)
graph/llm_factory.py     — LLM provider selection (Gemini/OpenAI/Mistral)
graph/state.py           — GraphState TypedDict
graph/consts.py          — node name constants, env_bool()
ingestion.py             — PDF/XML loading, chunking, Chroma population
ingestion_fetch.py       — URL fetch logic for retsinformation.dk and IAEA
build_document_sources.py — builds document_sources.yaml from local PDFs
document_updates.py      — checks for newer versions (retsinformation.dk, IAEA)
eval/                    — RAGAS evaluation (run_eval.py, golden.json)
tests/                   — pytest suite
frontend/src/App.tsx     — main UI component
frontend/src/constants.ts — API URLs, configuration
```

### Adding a new node

1. Create file in `graph/nodes/`, implement function `(state: GraphState) -> dict`
2. Export from `graph/nodes/__init__.py`
3. Register in `graph/graph.py` with `workflow.add_node(NAME, fn)` and edges
4. Add constant in `graph/consts.py`

### Adding a new chain

1. Create file in `graph/chains/`, implement `get_*` factory function
2. Export from `graph/chains/__init__.py`

---

## 3. Development Conventions

- **Python**: `uv` for dependencies, `uv run pytest tests/ -v` for tests
- **Frontend**: `npm -C frontend run test`, `npm -C frontend run build`
- **Linting**: pre-commit hooks (`.pre-commit-config.yaml`)
- **Environment variables**: always update `.env.example` when adding new variables
- **Chroma collections**: `radiation-iaea` and `radiation-dk-law` — do not rename without re-ingestion
- **Admin routes**: require `X-Admin-Token` header; without `ADMIN_TOKEN` → 503

---

## 4. Logseq Second Brain

Research knowledge lives in a separate repo (`eikrad/second-brain`) and is maintained independently from this project. Open that repo in its own editor window for ingestion. **Never write or update Logseq pages from within this project's context** — doing so will color the knowledge with project-specific framing.

This project only *reads* from the second brain.

### Rule: query before decide

Before any architectural decision (new node, retrieval strategy, embedding change):
1. `find_pages_by_property topic=rag-architecture` or `search` for the concept
2. `get_page_content` of relevant pages
3. Decide with cited sources from the graph — not from training data alone

Before answering a complex radiation safety question:
1. `find_pages_by_property topic=<topic>` or Datalog `query`
2. `get_page_content` + `get_page_backlinks` for related concepts
3. Answer with references to Logseq pages

### Query workflow

1. **`find_pages_by_property`** — fast property filter by `topic` or `document-id`
2. **`query`** — Datalog for precise combination search
3. **`get_page_content`** — load only relevant pages (keep context small)
4. **`get_page_backlinks`** — traverse related concepts via graph if needed
5. **`search`** — full-text fallback

Goal: load as few pages as needed, then synthesize in the LLM.

---

## 5. Documents in this project (already ingested)

These PDFs are stored locally and already ingested into Chroma:

**IAEA Standards:**
GSR-1, GSR-2, GSR-3, GSR-4, GSR-5, GSR-6, GSR-7,
SSG-11, SSG-39, SSG-40, SSG-44, SSG-46, SSG-86, SSG-87,
SSR-6, TECDOC-1380, TECDOC-1638, nuclear_safety_measures (24G)

**Danish sources (Bekendtgørelse):**
BEK-2025-138405, BEK-2025-138505, Brug af åbne radioaktive kilder,
Udarbejdelse af en sikkerhedsvurdering

These should be the first pages created in Logseq under `Sources/IAEA/` and `Sources/Danish/`.
