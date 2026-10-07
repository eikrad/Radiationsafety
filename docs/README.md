# docs/

Documentation for the Radiation Safety RAG system.

## Contents

| File | What it covers |
|------|----------------|
| [architecture.md](architecture.md) | RAG pipeline nodes, chains, ingestion workflow, LLM providers (incl. Ollama / privacy mode), API routes — with Mermaid diagrams |
| [production-readiness.md](production-readiness.md) | Security, admin auth, rate limiting, observability, container hardening, and a runbook |
| [maintenance.md](maintenance.md) | Dated log of periodic dependency checks, upgrade notes and security findings (newest first) |
| [releasing.md](releasing.md) | release-please flow, commit types and version bumps, how to merge PRs |

Related files outside `docs/`: [eval/README.md](../eval/README.md) (evaluation harness), [frontend/README.md](../frontend/README.md) (UI structure), [CONTRIBUTING.md](../CONTRIBUTING.md), [ROADMAP.md](../ROADMAP.md), and `AGENTS.md` / `CLAUDE.md` (conventions for AI agents).

## Where to start

- **New to the project?** Start with [architecture.md](architecture.md) to understand how queries flow from the browser to the vector database and back.
- **Want to run locally without any API keys?** See [architecture.md — Privacy Mode](architecture.md#privacy-mode-ollama) for Ollama setup.
- **Deploying or operating the system?** Read [production-readiness.md](production-readiness.md) for container hardening, admin authentication, rate limiting configuration, and runbook entries.
- **Adding a new pipeline node or chain?** Follow the step-by-step guide at the bottom of [architecture.md](architecture.md#adding-a-new-node).
- **Maintaining dependencies?** See [maintenance.md](maintenance.md). For document updates, see the [Updating documents](architecture.md#updating-documents) section of architecture.md.
- **Cutting a release?** See [releasing.md](releasing.md).
- **Looking for the big picture?** The [README.md](../README.md) in the repo root has a quick-start guide, Docker setup, evaluation instructions, and links back here.
