#!/usr/bin/env bash
# Run this once after cloning eikrad/second-brain locally.
# Usage: bash scripts/setup_second_brain.sh /path/to/second-brain
set -e

TARGET="${1:-$HOME/second-brain}"

if [ ! -d "$TARGET/.git" ]; then
  echo "Error: $TARGET is not a git repo. Clone it first:"
  echo "  git clone https://github.com/eikrad/second-brain.git $TARGET"
  exit 1
fi

cd "$TARGET"
mkdir -p pages logseq journals assets

# --- logseq/config.edn ---
cat > logseq/config.edn << 'HEREDOC'
{:meta/version 1
 :preferred-format :markdown
 :pages-directory "pages"
 :journals-directory "journals"
 :feature/enable-journals? false
 :ui/enable-tooltip? true
 :graph/settings {:hidden-namespaces []}}
HEREDOC

# --- .gitignore ---
cat > .gitignore << 'HEREDOC'
logseq/bak/
logseq/version-files/
logseq/.recycle
.DS_Store
HEREDOC

# --- AGENTS.md ---
cat > AGENTS.md << 'HEREDOC'
# AGENTS.md — Second Brain

This repo is a Logseq knowledge base for radiation safety research and RAG/AI architecture. It serves as a persistent, compiled knowledge source for all AI coding agents.

---

## Purpose

- **Domain knowledge**: IAEA standards, Danish regulations — grounded answers without hallucination
- **Architecture research**: RAG patterns, LangGraph design — informed coding decisions
- **Project-independent**: shared across multiple projects and tools (Claude Code, Cursor, Codex, Gemini CLI)

---

## Rule: query before decide

Before any architectural decision (new node, retrieval strategy, embedding change):
1. `find_pages_by_property topic=rag-architecture` or `search` for the concept
2. `get_page_content` of relevant pages
3. Decide with cited sources from the graph — not from training data alone

Before answering a radiation safety question:
1. `find_pages_by_property topic=<topic>` or Datalog `query`
2. `get_page_content` + `get_page_backlinks` for related concepts
3. Answer with references to Logseq pages

---

## Namespaces

```
Sources/IAEA/          — IAEA standards (GSR, SSG, SSR, TECDOC)
Sources/Danish/        — Danish Bekendtgørelser
Sources/Architecture/  — RAG patterns, LangGraph, embedding strategies
Sources/Research/      — Academic papers
Sources/Other/         — Podcasts, blogs
Concepts/              — Key terms (ALARA, Dosimetry, RAG, ...)
Regulations/           — Regulatory frameworks
Index                  — Master index (keep up to date)
```

---

## Properties (required for every new page)

```
source-type::   iaea-standard | iaea-tecdoc | danish-law | paper | book | podcast | other
document-id::   e.g. GSR-3, SSG-46, BEK-2025-138405
topic::         dosimetry | transport | medical | research | waste | emergency | regulatory | rag-architecture
language::      en | da | de
status::        ingested | reviewed | needs-update | superseded
date::          YYYY-MM-DD
url::           (optional)
```

---

## Ingest workflow (adding a new page)

1. `create_page` — namespace + properties + content (summary, key points, links)
2. `query` — find related pages (same topic)
3. `update_page` — add back-links to 3–5 related pages (append mode)
4. `update_page` — update [[Index]]

## Query workflow (fetching context)

1. `find_pages_by_property` — fast property filter
2. `query` — Datalog for combination search
3. `get_page_content` — load only relevant pages
4. `get_page_backlinks` — traverse graph if needed
5. `search` — full-text fallback

---

## Filename convention

Logseq encodes `/` in page titles as `___` (triple underscore) in filenames:

```
Sources/IAEA/GSR-3  →  pages/Sources___IAEA___GSR-3.md
Concepts/ALARA      →  pages/Concepts___ALARA.md
```

---

## MCP configuration (local)

```json
{
  "mcpServers": {
    "logseq": {
      "command": "npx",
      "args": ["-y", "mcp-logseq"],
      "env": {
        "LOGSEQ_GRAPH_PATH": "/absolute/path/to/second-brain"
      }
    }
  }
}
```

- Claude Code: `~/.claude/claude_desktop_config.json`
- Cursor: `.cursor/mcp.json` (project or global)
- Codex: `~/.codex/config.json`
HEREDOC

# --- pages/Index.md ---
cat > pages/Index.md << 'HEREDOC'
## Second Brain — Master Index

> Compiled knowledge base for radiation safety research and RAG/AI architecture.
> Raw sources stay immutable; this wiki is LLM-maintained.
> Follow the schema in [[AGENTS]] when adding or updating pages.

---

## Sources

### IAEA Standards
| Page | ID | Topic | Status |
|---|---|---|---|
| [[Sources/IAEA/GSR-1]] | GSR-1 | regulatory | ingested |
| [[Sources/IAEA/GSR-2]] | GSR-2 | regulatory | ingested |
| [[Sources/IAEA/GSR-3]] | GSR-3 | dosimetry, regulatory | ingested |
| [[Sources/IAEA/GSR-4]] | GSR-4 | regulatory | ingested |
| [[Sources/IAEA/GSR-5]] | GSR-5 | waste | ingested |
| [[Sources/IAEA/GSR-6]] | GSR-6 | waste | ingested |
| [[Sources/IAEA/GSR-7]] | GSR-7 | emergency | ingested |
| [[Sources/IAEA/SSG-11]] | SSG-11 | research | ingested |
| [[Sources/IAEA/SSG-39]] | SSG-39 | waste | ingested |
| [[Sources/IAEA/SSG-40]] | SSG-40 | waste | ingested |
| [[Sources/IAEA/SSG-44]] | SSG-44 | regulatory | ingested |
| [[Sources/IAEA/SSG-46]] | SSG-46 | medical | ingested |
| [[Sources/IAEA/SSG-86]] | SSG-86 | transport | ingested |
| [[Sources/IAEA/SSG-87]] | SSG-87 | research | ingested |
| [[Sources/IAEA/SSR-6]] | SSR-6 | transport | ingested |
| [[Sources/IAEA/TECDOC-1380]] | TECDOC-1380 | dosimetry | ingested |
| [[Sources/IAEA/TECDOC-1638]] | TECDOC-1638 | dosimetry | ingested |

### Danish Sources
| Page | ID | Topic | Status |
|---|---|---|---|
| [[Sources/Danish/BEK-2025-138405]] | BEK-2025-138405 | regulatory | ingested |
| [[Sources/Danish/BEK-2025-138505]] | BEK-2025-138505 | regulatory | ingested |

### Architecture & Research
| Page | Topic |
|---|---|
| [[Sources/Architecture/RAG-Patterns]] | Retrieval strategies, chunking, reranking |
| [[Sources/Architecture/LangGraph-Patterns]] | LangGraph workflow design |

---

## Concepts
- [[Concepts/ALARA]]
- [[Concepts/Dosimetry]]
- [[Concepts/Contamination]]
- [[Concepts/Transport-Safety]]
- [[Concepts/RAG]]

## Regulations
- [[Regulations/BSS-Framework]]
- [[Regulations/Danish-Regulatory-Framework]]
HEREDOC

# --- pages/Sources___IAEA___GSR-3.md ---
cat > "pages/Sources___IAEA___GSR-3.md" << 'HEREDOC'
source-type:: iaea-standard
document-id:: GSR-3
topic:: dosimetry, regulatory
language:: en
status:: ingested
date:: 2014-01-01
url:: https://www.iaea.org/publications/8930/radiation-protection-and-safety-of-radiation-sources-international-basic-safety-standards

## Radiation Protection and Safety of Radiation Sources — International Basic Safety Standards

The central IAEA document establishing dose limits, ALARA requirements, and responsibilities for radiation protection across all practices.

### Key Provisions
- Occupational dose limit: **20 mSv/year** averaged over 5 years, max 50 mSv in any single year
- Public dose limit: **1 mSv/year**
- [[Concepts/ALARA]] principle mandated for all practices
- Defines *practices* vs *interventions* distinction
- Covers medical, industrial, research, transport, and emergency contexts

### Structure
- Part I: General requirements (all practices)
- Part II: Occupational exposure
- Part III: Public exposure
- Part IV: Medical exposure
- Part V: Potential exposure and safety assessment

### Relation to other documents
- Supersedes Safety Series No. 115 (1996)
- Implemented in Denmark via [[Sources/Danish/BEK-2025-138405]]
- Medical specifics detailed in [[Sources/IAEA/SSG-46]]
- Transport specifics in [[Sources/IAEA/SSR-6]]
- Emergency response in [[Sources/IAEA/GSR-7]]
- Occupational protection guide: [[Sources/IAEA/SSG-87]]
HEREDOC

# --- pages/Sources___Architecture___RAG-Patterns.md ---
cat > "pages/Sources___Architecture___RAG-Patterns.md" << 'HEREDOC'
source-type:: other
topic:: rag-architecture
language:: en
status:: needs-update

## RAG Architecture Patterns

Current state of retrieval-augmented generation patterns relevant to this project.

### Chunking Strategies
- **Fixed-size**: simple, fast, loses semantic boundaries
- **Semantic chunking**: split on topic shifts — better recall, higher cost
- **Contextual chunking** (Anthropic, 2024): prepend chunk-level context before embedding → significant recall improvement

### Retrieval Strategies
- **Dense retrieval** (current): Gemini embeddings + cosine similarity in Chroma
- **Sparse (BM25)**: keyword overlap, good for exact terms (e.g. regulation IDs like "GSR-3")
- **Hybrid**: dense + sparse combined via RRF — best of both, worth evaluating

### Reranking
- **Cross-encoder reranking**: rerank top-k with a separate model before generation — improves precision
- Tradeoff: +latency, +cost, +relevance
- Relevant when retrieval returns many marginally relevant chunks

### Query Transformation
- **HyDE** (Hypothetical Document Embeddings): generate a hypothetical answer, embed that for retrieval
- **Step-back prompting**: broaden the query before retrieval for abstract questions
- Currently used in this project: `search_query_chain.py` reformulates the query

### Graph RAG
- Microsoft GraphRAG (2024): build knowledge graph from corpus, traverse for complex multi-hop queries
- Relevant for linking IAEA standards to Danish law (regulatory chains)
- High setup cost; consider if multi-hop queries become a bottleneck

### Relevant for this project
- Sparse/hybrid retrieval would help with exact regulation ID lookups (e.g. "BEK-138405")
- Contextual chunking worth testing given long, structured PDF documents
- See [[Sources/Architecture/LangGraph-Patterns]] for workflow-level patterns
HEREDOC

# --- pages/Sources___Architecture___LangGraph-Patterns.md ---
cat > "pages/Sources___Architecture___LangGraph-Patterns.md" << 'HEREDOC'
source-type:: other
topic:: rag-architecture
language:: en
status:: ingested

## LangGraph Workflow Patterns

Design patterns for LangGraph-based RAG workflows, relevant to `graph/graph.py`.

### Core Patterns

**Corrective RAG (CRAG)**
- Retrieve → Grade documents → if insufficient: web search or re-retrieve
- Current project implements this: `grade_documents` → `decide_to_generate` → `web_search`

**Self-RAG**
- LLM decides at each step whether to retrieve, and grades its own output
- Current project: `grade_generation_grounded` implements the grading loop

**Adaptive RAG**
- Route queries to different retrieval strategies based on query type
- Extension point: could route "regulation lookup" to BM25 vs "conceptual question" to dense

### State Management
- `GraphState` TypedDict as single source of truth — good pattern, keep flat
- Avoid nested state; LangGraph's checkpointing works best with flat dicts
- Current: `graph/state.py`

### Retry Loops
- Cap retries explicitly (current: `retrieval_count < 3`) to prevent infinite loops
- Always have a terminal fallback (current: web search → verify_trusted → finalize)

### Parallelism
- `Send` API for fan-out: retrieve from multiple collections simultaneously
- Worth evaluating: parallel retrieval from `radiation-iaea` and `radiation-dk-law`

### Streaming
- LangGraph supports token streaming via `.astream_events()`
- Current project uses sync; streaming would improve UX for long answers

### Observability
- LangSmith tracing already wired (`LANGCHAIN_API_KEY`)
- Add node-level metadata tags for easier trace filtering

### See also
- [[Sources/Architecture/RAG-Patterns]] for retrieval-level patterns
- `graph/graph.py` in Radiationsafety repo for current implementation
HEREDOC

# --- pages/Concepts___ALARA.md ---
cat > "pages/Concepts___ALARA.md" << 'HEREDOC'
source-type:: other
topic:: dosimetry, regulatory
language:: en
status:: ingested

## ALARA — As Low As Reasonably Achievable

Core principle of radiation protection: doses should be kept as low as reasonably achievable, taking economic and social factors into account.

### Definition
Mandated by [[Sources/IAEA/GSR-3]] (Requirement 5) for all planned exposure situations.
Not a dose limit — a process obligation to optimize protection.

### Three pillars of optimization
- **Time**: minimize time spent near radiation source
- **Distance**: maximize distance (dose ∝ 1/r²)
- **Shielding**: use appropriate shielding materials

### Regulatory basis
- IAEA GSR-3 §3.10–3.14
- Danish implementation: [[Sources/Danish/BEK-2025-138405]]
- Medical context: [[Sources/IAEA/SSG-46]] — ALARA in diagnostic imaging

### Related concepts
- [[Concepts/Dosimetry]] — how doses are measured
- [[Concepts/Contamination]] — ALARA applies to contamination control too
HEREDOC

# Commit and push
git add -A
git commit -m "Initial Logseq second brain structure

Namespaces: Sources/IAEA, Sources/Danish, Sources/Architecture, Concepts.
Seed pages: Index, GSR-3, RAG-Patterns, LangGraph-Patterns, ALARA.
AGENTS.md with schema, property conventions, ingest/query workflows,
filename convention, and MCP config reference."

git push -u origin main

echo ""
echo "Done! second-brain pushed to GitHub."
echo ""
echo "Next: configure mcp-logseq in your tool of choice:"
echo "  LOGSEQ_GRAPH_PATH=$(pwd)"
