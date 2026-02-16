# Radiation Safety RAG – Project Reference for Agents

This document describes the project structure, data flow, and conventions. Read it when you need to navigate the codebase or add changes.

## Overview

RAG system over IAEA and Danish radiation-safety documents. Stack: **LangGraph** (orchestration), **Chroma** (vector store), **FastAPI** (backend), **React** (frontend). Optional **Brave Search** fallback when retrieval is weak. LLM: **Mistral**, **Gemini**, or **OpenAI** (selectable in the UI; API keys from frontend Settings or server `.env`).

## Directory Structure

```
Radiationsafety/
├── api/
│   └── main.py              # FastAPI app: /query, /health, static frontend
├── graph/
│   ├── graph.py             # LangGraph workflow definition and routing
│   ├── state.py             # GraphState TypedDict
│   ├── consts.py            # Node names (RETRIEVE, GRADE_DOCUMENTS, …)
│   ├── llm_factory.py       # get_llm(provider, api_key, model_variant)
│   ├── nodes/
│   │   ├── retrieve.py      # Chroma retrieval (IAEA + DK)
│   │   ├── grade_documents.py  # Retrieval grader per doc
│   │   ├── generate.py      # RAG generation with chat history
│   │   ├── web_search.py    # Brave Search fallback
│   │   └── __init__.py
│   └── chains/
│       ├── generation.py       # get_generation_chain(llm)
│       ├── retrieval_grader.py  # get_retrieval_grader(llm)
│       ├── answer_grader.py     # get_answer_grader(llm)
│       └── hallucinations_grader.py  # get_hallucination_grader(llm)
├── frontend/
│   └── src/
│       ├── App.tsx          # Main app, API calls, model/settings state
│       ├── App.css
│       ├── main.tsx
│       ├── constants.ts    # MODELS, STORAGE_KEYS, MODEL_VARIANTS
│       ├── types.ts
│       └── components/
│           ├── QueryForm.tsx
│           ├── ResponseDisplay.tsx
│           ├── ModelSelector.tsx
│           └── SettingsModal.tsx
├── ingestion.py            # PDF load (local + from URLs), chunk, embed; Chroma collections
├── ingestion_fetch.py      # Fetch PDFs from URLs (Retsinformation, IAEA, direct)
├── main.py                 # CLI entry (optional)
├── tests/
│   ├── conftest.py         # pytest fixtures, mock graph, env
│   ├── test_api.py
│   ├── test_graph.py
│   ├── test_llm_factory.py
│   └── test_ingestion.py
├── .env.example
├── pyproject.toml
└── README.md
```

## Data Flow

1. **Request**: Frontend sends `POST /api/query` with `question`, `chat_history`, `model`, optional `model_variant`, optional `api_keys`.
2. **API** (`api/main.py`): Validates input, resolves model + API key, builds LLM. Short-circuits for non-questions (e.g. "Thank you"). If user sends `api_keys`, LangSmith tracing is disabled for that request. Calls `graph.invoke(initial_state)` with `llm` in state.
3. **Graph** (`graph/graph.py`):
   - **retrieve** → Chroma (IAEA + DK), merges docs.
   - **grade_documents** → Retrieval grader per doc; sets `web_search` if none relevant.
   - **Conditional**: if `web_search` and `WEB_SEARCH_ENABLED` → **web_search** (Brave), else → **generate**.
   - **web_search** → adds web results to docs, then → **generate**.
   - **generate** → RAG answer with context + chat history; updates `chat_history` in state.
   - **grade_generation_grounded** (conditional): hallucination + answer grader → `useful` (END), `web_search` (retry Brave once), or `end` → **finalize**.
   - **finalize** → sets `retrieval_warning` if web search was used and results were poor → END.
4. **Response**: API maps graph result to `QueryResponse` (answer, sources, chat_history, warning).

## Key Conventions

- **New graph node**: Add function in `graph/nodes/`, export in `graph/nodes/__init__.py`, register in `graph/graph.py` (add_node, edges).
- **New chain / LLM usage**: Prefer factory `get_*_chain(llm)` or `get_*_grader(llm)` in `graph/chains/` so the graph can pass per-request `llm` from state.
- **Env**: `.env` from `.env.example`. Important: `LLM_PROVIDER`, `GOOGLE_API_KEY` / `MISTRAL_API_KEY` / `OPENAI_API_KEY`, `WEB_SEARCH_ENABLED`, `BRAVE_SEARCH_API_KEY`, `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY`.
- **Frontend**: Model and API keys in localStorage; keys sent only in request body. Settings UI: API key inputs + per-provider model variant (e.g. Gemini Flash-Lite vs Pro).
- **Tests**: Backend `uv run pytest tests/ -v` (conftest mocks graph, sets TESTING=true). Frontend `npm run test` (Vitest).

## Collections (Chroma)

- `radiation-iaea`: IAEA PDFs
- `radiation-dk-law`: Danish legislation PDFs (Bekendtgørelse)

Embeddings use `LLM_PROVIDER` (Mistral or Gemini) in `ingestion.py` and `graph/llm_factory.py` (get_embeddings). Chat LLM is independent and can be overridden per request from the frontend.

## Document updates

- **Danish (retsinformation.dk)**: No API. URLs are resolved via Brave Search (`site:retsinformation.dk`) and/or probing `eli/lta/YEAR/nr`; then the document page is fetched and "Senere ændringer til forskriften" is parsed to get the newest consolidated version (e.g. BEK 670 → 1385). **SST (sst.dk)**: Two Danish sources (e.g. "Brug af åbne radioaktive kilder", "Udarbejdelse af en sikkerhedsvurdering") are vejledninger on sst.dk; URLs are resolved via Brave Search `site:sst.dk` and downloaded as PDF (not XML).
- **IAEA (iaea.org)**: Publication pages can show **"Superseded by: …"**. GET the page, parse for "Superseded by"; if present, use the superseding title and link as the newer version and download target.

## Credits

Patterns and inspiration from **Eden Marco**’s LangChain/LangGraph course and repo: [github.com/emarco177/langchain-course](https://github.com/emarco177/langchain-course) (Apache-2.0). When modifying or reusing ideas from that repo, respect its license and attribution requirements.
