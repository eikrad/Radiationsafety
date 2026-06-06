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
graph/nodes/             — retrieve, grade_documents, grade_generation, retrieve_missing, generate, web_search, verify_trusted
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
2. Import directly from the chain file in the node(s) that use it (e.g. `from graph.chains.my_chain import get_my_chain`). Note: `graph/chains/__init__.py` is intentionally minimal; chains are not re-exported there.

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

This project uses **Logseq** (via `mcp-logseq`) as a compiled knowledge base for radiation safety research and RAG architecture. The concept follows Karpathy's LLM wiki principle: raw sources stay immutable, the LLM maintains linked pages in Logseq.

The second brain lives in a separate repo: `eikrad/second-brain`. Configure `mcp-logseq` to point at your local clone.

### Rule: query before decide

Before any architectural decision (new node, retrieval strategy, embedding change):
1. `find_pages_by_property topic=rag-architecture` or `search` for the concept
2. `get_page_content` of relevant pages
3. Decide with cited sources from the graph — not from training data alone

Before answering a complex radiation safety question:
1. `find_pages_by_property topic=<topic>` or Datalog `query`
2. `get_page_content` + `get_page_backlinks` for related concepts
3. Answer with references to Logseq pages

### Namespaces

```
Sources/IAEA/          — IAEA standards and safety guides (GSR, SSG, SSR, TECDOC)
Sources/Danish/        — Danish Bekendtgørelser (retsinformation.dk)
Sources/Architecture/  — RAG patterns, LangGraph, embedding strategies
Sources/Research/      — Academic papers, external studies
Sources/Other/         — Podcasts, blogs, misc
Concepts/              — Key terms (Dosimetry, ALARA, Contamination, RAG, ...)
Regulations/           — Regulatory frameworks and comparisons
Index                  — Master index of all pages
```

### Page properties

Every new page must have these properties:

```
source-type::   iaea-standard | iaea-tecdoc | danish-law | paper | book | podcast | other
document-id::   e.g. GSR-3, SSG-46, BEK-2025-138405, TECDOC-1380
topic::         dosimetry | transport | medical | research | waste | emergency | regulatory | rag-architecture
language::      en | da | de
status::        ingested | reviewed | needs-update | superseded
date::          YYYY-MM-DD
url::           (optional)
```

### Ingest workflow (new source)

When a new source arrives (PDF, article, podcast note):

1. **`create_page`** — create page under the appropriate namespace
   - Title: `Sources/IAEA/GSR-3` or `Concepts/ALARA`
   - Set properties as above
   - Content: summary, key points, relevant paragraphs as blocks

2. **`query`** — find related pages:
   ```clojure
   [:find (pull ?p [:block/name])
    :where [?p :block/properties ?props]
           [(get ?props :topic) ?t]
           [(= ?t "dosimetry")]]
   ```

3. **`update_page`** — update 3–7 related pages with back-links and cross-references (append mode)

4. **`update_page`** — add entry to `Index` page

### Query workflow (fetching context)

1. **`find_pages_by_property`** — fast property filter by `topic` or `document-id`
2. **`query`** — Datalog for precise combination search (e.g. topic=transport AND language=en)
3. **`get_page_content`** — load only relevant pages (keep context small)
4. **`get_page_backlinks`** — traverse related concepts via graph if needed
5. **`search`** — full-text fallback when properties are insufficient

Goal: load as few pages as needed, then synthesize in the LLM.

### Lint routine (periodic)

1. **`query`** — find pages missing `source-type` property
2. **`query`** — review pages with `status: needs-update`
3. **`get_page_backlinks`** — identify orphan pages (no incoming links)
4. Mark outdated documents with `status: superseded` when a newer version exists

### Logseq MCP tools reference

| Tool | When |
|---|---|
| `create_page` | Ingest a new source |
| `update_page` | Add cross-references (append) or correct content (replace) |
| `update_block` | Edit a single block by UUID |
| `find_pages_by_property` | Fast property filter (topic, status, document-id) |
| `query` | Complex Datalog queries |
| `get_page_backlinks` | Which pages link to X? |
| `get_pages_from_namespace` | All pages under Sources/IAEA/ |
| `search` | Full-text fallback |

### Example Datalog queries

```clojure
;; All IAEA standards on transport
[:find (pull ?p [:block/name :block/properties])
 :where [?p :block/properties ?props]
        [(get ?props :source-type) "iaea-standard"]
        [(get ?props :topic) "transport"]]

;; All pages that need review
[:find (pull ?p [:block/name])
 :where [?p :block/properties ?props]
        [(get ?props :status) "needs-update"]]

;; All Danish-language sources
[:find (pull ?p [:block/name])
 :where [?p :block/properties ?props]
        [(get ?props :language) "da"]]
```

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
