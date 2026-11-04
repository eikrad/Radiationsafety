# Documentation index

Documentation for the Radiation Safety RAG system. Start with the [root README](../README.md) for what the project is and how to run it.

## Contents

| File | Kind | What it covers |
|------|------|----------------|
| [architecture.md](architecture.md) | Reference | Request flow, LangGraph nodes and chains, state, ingestion and document-update workflows, LLM providers (incl. Ollama privacy mode), API routes, how to add a node or chain |
| [production-readiness.md](production-readiness.md) | Reference / runbook | Route exposure and admin auth, rate limiting, metrics, container hardening, CI safeguards, runbook |
| [releasing.md](releasing.md) | How-to | release-please flow, commit types and version bumps, how to merge each kind of PR |
| [maintenance.md](maintenance.md) | Log | Dated log of periodic dependency checks, upgrade notes and security findings (newest first) |
| [privacy-mode-docker-plan.md](privacy-mode-docker-plan.md) | Plan (not started) | Optional Ollama container (`--profile ollama`) so privacy mode works under Docker |
| [handoff-held-out-test-run.md](handoff-held-out-test-run.md) | Protocol | Run plan for the held-out test set (#167): baselines, tune/confirm rules, experiments, frontier comparison |
| [handoff-talk-prep.md](handoff-talk-prep.md) | Record | Open measurements planned for the 2026-10-30 talk (local model on `confirm`, loop analysis, language vs. structure) |

Related files outside `docs/`: [eval/README.md](../eval/README.md) (evaluation harness), [frontend/README.md](../frontend/README.md) (UI structure), [CONTRIBUTING.md](../CONTRIBUTING.md), [ROADMAP.md](../ROADMAP.md) (plans and measured results), [CHANGELOG.md](../CHANGELOG.md), and `AGENTS.md` / `CLAUDE.md` (conventions for AI agents).

## Where to start

- **New to the project?** [architecture.md](architecture.md) shows how a question flows from the browser to the vector database and back.
- **Running without cloud keys?** [architecture.md, Privacy Mode](architecture.md#privacy-mode-ollama) (Ollama); setup steps are in the [README](../README.md#privacy-mode-fully-local).
- **Deploying or operating?** [production-readiness.md](production-readiness.md).
- **Adding a pipeline node or chain?** [architecture.md, Adding a new node](architecture.md#adding-a-new-node).
- **Updating source documents?** [architecture.md, Updating documents](architecture.md#updating-documents).
- **Cutting a release?** [releasing.md](releasing.md).
- **Checking dependencies?** [maintenance.md](maintenance.md).
- **Measuring answer quality?** [eval/README.md](../eval/README.md).
