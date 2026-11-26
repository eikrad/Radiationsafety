# Contributing

Thanks for your interest in contributing to Radiation Safety RAG.

## Development setup

1. **Clone and install**
   - `uv sync` (Python)
   - `npm ci` in `frontend/` for the UI.

2. **Environment**
   - Copy `.env.example` to `.env`. Set **`SCW_SECRET_KEY`** (Scaleway: default for answers and embeddings). To use another provider, set `LLM_PROVIDER` / `EMBEDDING_PROVIDER` and the matching key (e.g. `GOOGLE_API_KEY`, `OPENAI_API_KEY`).

3. **Document registry**
   - `document_sources.yaml` holds source URLs for ingestion and “Check for updates”. Prefer **generating** it (`uv run python build_document_sources.py`, see [Managing documents](README.md#managing-documents)) or copying from `document_sources.example.yaml` rather than committing repo-specific URLs. It is listed in `.gitignore` by default; remove that line if you want to commit a shared registry.

## Running tests

- **Backend:** `uv run pytest tests/ -v`
- **Frontend (unit):** `cd frontend && npm run test` (or `npm run test:watch` for watch mode)
- **Frontend (E2E):** `cd frontend && npx playwright install --with-deps chromium && npm run test:e2e`: UI-only Playwright tests against a mocked API (`frontend/e2e/`)
- **Frontend (visual):** `cd frontend && npm run test:visual:docker`: screenshot comparisons in light and dark mode (`frontend/visual/`), run in the Playwright Docker image so fonts match CI. After an intended UI change, refresh the baselines with `npm run test:visual:update` and commit the new PNGs
- **Coverage:** `uv run pytest tests/ --cov` and `cd frontend && npm run test:coverage`

CI runs these (plus E2E, visual and Docker checks) on push and on pull requests.

## Code quality

- **Formatting:** `black .` and `isort .` (config in `pyproject.toml`).
- **Linting / type checking:** `uv run ruff check .` for the whole codebase. `mypy` is intentionally scoped in CI to `api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip`; run the same command locally before submitting changes to those files.
- **Pre-commit hook (recommended):**
  - Install once per clone: `uv run pre-commit install`
  - Run manually on all files: `uv run pre-commit run --all-files`
  - Hook config lives in `.pre-commit-config.yaml` and currently enforces `black --check` and `isort --check-only`.

## Documentation

- **Architecture diagram:** `architecture.svg` / `architecture.png` in the repo root are generated from `architecture.mmd` (linked from [docs/architecture.md](docs/architecture.md)). If you change the RAG graph (nodes or flow in `graph/`), update `architecture.mmd` and regenerate the image: `uv run python scripts/render_architecture.py`, then commit the updated SVG.

## Pull requests

- Branch from and open your PR against **`staging`** — never against `master` directly. `staging` is the integration/QA gate; `master` only updates by merging `staging` in after validation.
- Ensure tests and lint pass (CI will run them).
- Keep changes focused; mention any env or setup requirements in the PR description.


