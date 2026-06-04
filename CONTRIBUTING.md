# Contributing

Thanks for your interest in contributing to Radiation Safety RAG.

## Development setup

1. **Clone and install**
   - `uv sync` (Python)
   - `npm ci` in `frontend/` for the UI.

2. **Environment**
   - Copy `.env.example` to `.env`. Set **`GOOGLE_API_KEY`** for ingestion and retrieval (required). Set `LLM_PROVIDER` and the matching key (e.g. `OPENAI_API_KEY`) if you run the app or eval with a specific model for generation.

3. **Document registry**
   - `document_sources.yaml` holds source URLs for ingestion and “Check for updates”. Prefer **generating** it (see [Building document_sources.yaml](README.md#building-document_sourcesyaml-from-local-pdfs)) or copying from `document_sources.example.yaml` rather than committing repo-specific URLs. It is listed in `.gitignore` by default; remove that line if you want to commit a shared registry.

## Running tests

- **Backend:** `uv run pytest tests/ -v`
- **Frontend:** `cd frontend && npm run test` (or `npm run test:watch` for watch mode)

CI runs both on push and on pull requests.

## Code quality

- **Formatting:** `black .` and `isort .` (config in `pyproject.toml`).
- **Linting / type checking:** `uv run ruff check .` and `uv run mypy .` (see `pyproject.toml`). Fix any reported issues before submitting.
- **Pre-commit hook (recommended):**
  - Install once per clone: `uv run pre-commit install`
  - Run manually on all files: `uv run pre-commit run --all-files`
  - Hook config lives in `.pre-commit-config.yaml` and currently enforces `black --check` and `isort --check-only`.

## Documentation

- **Architecture diagram:** The README shows `architecture.svg`, generated from `architecture.mmd`. If you change the RAG graph (nodes or flow in `graph/`), update `architecture.mmd` and regenerate the diagram: `uv run python scripts/render_architecture.py`, then commit the updated diagram files (`architecture.svg`, `architecture.png`).

## Pull requests

- Open a PR against the default branch. Ensure tests and lint pass (CI will run them).
- Keep changes focused; mention any env or setup requirements in the PR description.


