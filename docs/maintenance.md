# Maintenance Log

Weekly dependency and health checks for the Radiationsafety RAG project.

---

## 2026-07-08

### Checks performed
- Established baseline: `uv sync --all-extras`, `uv run pytest tests/ -v` (177 passed), `uv run ruff check .`, `uv run black --check .`, `uv run isort --check .`, `uv run mypy api/main.py api/rate_limit.py tests/test_api.py`, `uv run pre-commit run --all-files` — all green before and after changes
- `npm -C frontend ci`, `npm -C frontend run test` (29 passed), `npm -C frontend run lint`, `npm -C frontend run build` — all green before and after changes
- Python dependency audit: `uv export --format requirements-txt --no-hashes` + `pip-audit` (run in environment-scan mode against the synced `.venv`, since resolving the exported requirements file against the live PyPI index failed — this repo's lockfile already contains newer point releases than are currently published upstream, e.g. `numpy==2.5.0`)
- `uv pip list --outdated` to check newer versions of all resolved packages, direct and transitive
- Frontend: `npm -C frontend outdated`, `npm -C frontend audit`
- Reviewed `.github/workflows/ci.yml` for Node/Python version drift against `pyproject.toml`, `.python-version`, and `frontend/package.json`/`vite` engine requirements
- Reviewed `Dockerfile` and `frontend/Dockerfile` for version drift against the CI toolchain
- Verified `.github/dependabot.yml` and `.github/workflows/weekly-audit.yml` (added in a prior cycle, PR #43) are present and match the intended spec: dependabot covers npm (`/frontend`), pip (`/`), and github-actions (`/`) ecosystems, weekly, targeting `staging`; the weekly-audit workflow runs `pip-audit` (via `uv export`) and `npm audit` and opens/updates a `security-audit`-labelled issue on findings. No changes needed — this repo already has parity with `eikrad/Job-Tracker`.

### Fixes applied

- **Docker Node version drift** — `frontend/Dockerfile` was still pinned to `node:20-alpine`, while CI (`.github/workflows/ci.yml`) has used Node 22 since the 2026-06-10 cycle and Vite 8 targets `20.19+ or 22.12+`. Bumped to `node:22-alpine` so the container build stage matches the tested CI environment. (Docker daemon unavailable in this sandbox so the image itself could not be built end-to-end; the underlying `npm ci && npm run build` steps were verified separately on Node 22.)
- **Python dependency bumps** (patch/minor, no major version changes) — updated `pyproject.toml` lower bounds and re-locked `uv.lock`:
  - `docling` `>=2.101.0` → `>=2.111.0`
  - `langchain-google-genai` `>=4.2.0` → `>=4.2.7`
  - `langchain-mistralai` `>=1.1.0` → `>=1.1.6`
  - `langgraph` `>=1.2.0` → `>=1.2.8`
  - `uvicorn[standard]` `>=0.49.0` → `>=0.51.0`
  - `mypy` (dev) `>=2.1.0` → `>=2.2.0`

  Full test/lint suite re-verified green after each change. Deliberately used `uv lock --upgrade-package` per-package rather than a blanket `uv lock --upgrade`, because the latter also pulled `opencv-python` 4.13.0.92 → 5.0.0.93 (a major bump, transitive via `docling`/`rapidocr`) — left untouched, see below.
- **Frontend dependency bumps** (patch, within existing `package.json` semver ranges) — `npm -C frontend update`: `@typescript-eslint/eslint-plugin` / `@typescript-eslint/parser` `8.62.1` → `8.63.0`, `vite` `8.1.2` → `8.1.3`, `vitest` `4.1.9` → `4.1.10` (and assorted transitive patch bumps). `package.json` ranges (`^8.60.0`, `^8.0.0`, `^4.1.7`) unchanged; only `package-lock.json` updated.

### Security findings

- **CRITICAL — CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311 (ChromaDB "ChromaToast")** — pre-authentication remote code execution in `chromadb`'s `/api/v2/tenants/{tenant}/databases/{db}/collections` endpoint via a malicious embedding-function model reference with `trust_remote_code=true`. CVSS ~9.3–10. Affects all `chromadb` 1.0.0–1.5.9 (the current latest release on PyPI as of this cycle — **no patched version exists yet**, confirmed via `pip index versions chromadb` and the GitHub advisory). Flagged, not silently applied, because there is no safe version bump available at all (not even a major one).
  - **Risk assessment for this repo**: exposure appears low in practice. `ingestion.py` and `graph/` only ever construct `langchain_chroma.Chroma(...)` as an embedded/persistent client; the codebase never starts `chromadb`'s standalone HTTP server (`chroma run` / `chromadb.HttpClient`), and no application code sets `trust_remote_code=True` anywhere. The vulnerable `/api/v2/.../collections` server endpoint is therefore not exposed by this deployment as it stands today.
  - **Action**: no code change made. Recommend tracking upstream for a patched `chromadb` release and re-running `pip-audit` weekly (now automated via `weekly-audit.yml`); if `chromadb` is ever run as a standalone server or exposed over the network in the future, apply the documented interim mitigation (disable `trust_remote_code` server-side, restrict network access to the Chroma port) before doing so.
- `npm -C frontend audit`: 0 vulnerabilities found.

### Dependency status

**Python backend (`pyproject.toml`):**

| Package | Constraint | Status |
|---|---|---|
| `chromadb` | `>=1.5.0` | Current (1.5.9, latest on PyPI) — see security findings above |
| `fastapi` | `>=0.136.0` | Current (resolves to 0.139.0) |
| `langchain` | `>=1.3.0` | Current (resolves to 1.3.11) |
| `langgraph` | `>=1.2.8` | Updated this cycle |
| `docling` | `>=2.111.0` | Updated this cycle |
| `langchain-google-genai` | `>=4.2.7` | Updated this cycle |
| `langchain-mistralai` | `>=1.1.6` | Updated this cycle |
| `uvicorn` | `>=0.51.0` | Updated this cycle |
| `redis` | `>=8.0.0` | Current (resolves to 8.0.1) |
| `mypy` (dev) | `>=2.2.0` | Updated this cycle |

**Frontend (`frontend/package.json`):**

| Package | Constraint | Status |
|---|---|---|
| `react` | `^19.2.5` | Current |
| `react-markdown` | `^10.1.0` | Current |
| `vite` | `^8.0.0` | Current (resolves to 8.1.3, updated this cycle) |
| `typescript` | `~6.0.0` | Current within range — TypeScript 7 available, see below |
| `eslint` | `^10.0.0` | Current |
| `vitest` | `^4.1.7` | Current (resolves to 4.1.10, updated this cycle) |
| `@vitejs/plugin-react` | `^6.0.0` | Current |
| `globals` | `^16.5.0` | Upgrade candidate carried over from 2026-06-17 — see below |

### Major upgrades pending (manual review required)

| Package | Current | Available | Why not applied automatically |
|---|---|---|---|
| `chromadb` (Python) | `1.5.9` | — (no fix yet) | CRITICAL unpatched CVE-2026-45829; not a version-bump fix, tracked in Security findings above. No action possible until upstream ships a patch. |
| `typescript` (frontend) | `~6.0.0` (6.0.3) | `7.0.2` | Major version bump. TypeScript 7 ("Corsa") is a from-scratch native/Go-based compiler port with tooling and plugin-ecosystem implications — needs a dedicated review/testing cycle, not a drive-by bump. |
| `jsdom` (frontend, dev) | `^28.0.0` (28.1.0) | `29.1.1` | Major version bump; used only in the Vitest test environment but warrants its own verification pass. |
| `globals` (frontend, dev) | `^16.5.0` (16.5.0) | `17.7.0` | Major version bump, carried over from the 2026-06-17 cycle. ESLint 10 is compatible with both; still not applied without explicit review. |
| `eslint-plugin-react-refresh` (frontend, dev) | `^0.4.24` (0.4.26) | `0.5.3` | Pre-1.0 package — a `0.4` → `0.5` bump is outside the current caret range and can carry breaking changes under semver convention for `0.x` packages; requires a deliberate `package.json` range change rather than an automatic update. |

---

## 2026-06-17

### Checks performed
- Reviewed all CI workflows — all passing (last successful run: 2026-06-17T09:29:27Z on master)
- Reviewed `pyproject.toml` Python backend dependency bounds
- Reviewed `frontend/package.json` frontend dependencies
- Compared versions cross-repo against bandsearch-app

### Fixes applied
No fixes needed this cycle.

### Dependency status

**Python backend (`pyproject.toml`):**
Bounds updated as part of v0.4.0 (2026-06-11). All constraints reflect currently installed stable versions resolved by `uv.lock`.

| Package | Constraint | Status |
|---|---|---|
| `fastapi` | `>=0.136.0` | Current |
| `langchain` | `>=1.3.0` | Current |
| `langgraph` | `>=1.2.0` | Current |
| `chromadb` | `>=1.5.0` | Current |
| `uvicorn` | `>=0.49.0` | Current |
| `langchain-google-genai` | `>=4.2.0` | Current |
| `langchain-openai` | `>=1.3.0` | Current |
| `langchain-mistralai` | `>=1.1.0` | Current |
| `redis` | `>=8.0.0` | Current |
| `docling` | `>=2.101.0` | Current |

**Frontend (`frontend/package.json`):**

| Package | Constraint | Status |
|---|---|---|
| `react` | `^19.2.5` | Current |
| `react-markdown` | `^10.1.0` | Current |
| `vite` | `^8.0.0` | Current |
| `typescript` | `~6.0.0` | Current |
| `eslint` | `^10.0.0` | Current |
| `vitest` | `^4.1.7` | Current |
| `@vitejs/plugin-react` | `^6.0.0` | Current |
| `globals` | `^16.5.0` | Upgrade candidate (see below) |

### Upgrade candidates

| Package | In use | Available | Notes |
|---|---|---|---|
| `globals` (frontend) | `^16.5.0` | `^17.x` | ESLint 10 is compatible with both 16 and 17 — not blocking. bandsearch-app already on `^17.6.0`. Upgrade requires a `package-lock.json` refresh (`npm update globals`). |

---

## 2026-06-10

### Checks performed
- Reviewed all dependencies in `pyproject.toml` and `frontend/package.json`
- Reviewed CI workflow in `.github/workflows/ci.yml`
- Compared versions against current ecosystem state

### Fixes applied

- **CI Node version** — Bumped Node from `20` to `22` (LTS) in `.github/workflows/ci.yml`. Vite 8 officially targets Node 20.18+ or 22+; using the LTS release explicitly removes any ambiguity and aligns with the Node.js long-term support schedule.

### Dependency status

**Python backend (`pyproject.toml`):**
All deps use `>=` lower bounds and are resolved/pinned by `uv.lock`. No updates needed this cycle.

| Package | Constraint | Status |
|---|---|---|
| `fastapi` | `>=0.115.0` | Current |
| `langchain` | `>=1.2.7` | Current |
| `langgraph` | `>=1.0.7` | Current |
| `chromadb` | `>=1.4.1` | Current |
| `uvicorn` | `>=0.32.0` | Current |
| `langchain-google-genai` | `>=4.0.0` | Current |
| `langchain-openai` | `>=0.3.0` | Current |
| `langchain-mistralai` | `>=0.1.0` | Current |
| `redis` | `>=5.0.0` | Current |
| `docling` | `>=2.0.0` | Current |

**Frontend (`frontend/package.json`):**
All major deps already on latest major versions.

| Package | Constraint | Status |
|---|---|---|
| `react` | `^19.2.5` | Current |
| `react-markdown` | `^10.1.0` | Current |
| `vite` | `^8.0.0` | Current |
| `typescript` | `~6.0.0` | Current |
| `eslint` | `^10.0.0` | Current |
| `vitest` | `^4.1.7` | Current |
| `@vitejs/plugin-react` | `^6.0.0` | Current |

### No major upgrades pending
All packages are on current major versions this cycle.
