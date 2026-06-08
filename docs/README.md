# Documentation

Navigation guide for the `docs/` folder.

## Contents

| Path | What it covers |
|------|----------------|
| [architecture.md](architecture.md) | Full pipeline walkthrough — query path, each LangGraph node, chains, state fields, ingestion workflow, LLM provider selection, and API routes |
| [production-readiness.md](production-readiness.md) | Deployment reference — admin auth, rate limiting, container hardening, observability, and runbook for common issues |

## Where to start

- **New to the codebase?** Read [architecture.md](architecture.md) — it walks through the full query path from user question to cited answer, with Mermaid diagrams for each stage.
- **Setting up for production?** See [production-readiness.md](production-readiness.md) for auth, rate limits, Docker hardening, and a runbook.
- **Adding a new node or chain?** The step-by-step extension guide is at the bottom of [architecture.md](architecture.md).
- **Understanding document ingestion?** The ingestion pipeline and document update workflows are both diagrammed in [architecture.md](architecture.md).
