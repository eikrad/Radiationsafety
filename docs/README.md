# docs/

Documentation for the Radiation Safety RAG system.

## Contents

| File | What it covers |
|------|----------------|
| [architecture.md](architecture.md) | RAG pipeline nodes, chains, ingestion workflow, LLM providers, API routes — with Mermaid diagrams |
| [production-readiness.md](production-readiness.md) | Security, admin auth, rate limiting, observability, container hardening, and a runbook |
| [maintenance.md](maintenance.md) | Dependency versions, upgrade notes, and periodic maintenance tasks |

## Where to start

- **New to the project?** Start with [architecture.md](architecture.md) to understand how queries flow from the browser to the vector database and back.
- **Deploying or operating the system?** Read [production-readiness.md](production-readiness.md) for container hardening, admin authentication, rate limiting configuration, and runbook entries.
- **Adding a new pipeline node or chain?** Follow the step-by-step guide at the bottom of [architecture.md](architecture.md#adding-a-new-node).
- **Maintaining dependencies or documents?** See [maintenance.md](maintenance.md) for upgrade notes and document update procedures.
- **Looking for the big picture?** The [README.md](../README.md) in the repo root has a quick-start guide, Docker setup, evaluation instructions, and links back here.
