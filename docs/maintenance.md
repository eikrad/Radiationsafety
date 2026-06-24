# Maintenance Log

Weekly dependency and health checks for the Radiationsafety RAG project.

---

## 2026-06-24

### Checks performed
- Reviewed `pyproject.toml` (Python backend) and `frontend/package.json` (React/TypeScript)
- Reviewed `.github/workflows/ci.yml`
- Compared dependency constraints against ecosystem state

### Infrastructure added

- **Dependabot** — Added `.github/dependabot.yml` to automate weekly PR generation for:
  - Python (`pip`) — targets `staging`; LangChain-family packages grouped into a single PR
  - npm (`/frontend`) — targets `staging`
  - GitHub Actions — targets `staging`

  Dependabot also raises security alerts independently of scheduled version bumps, covering
  known CVEs in the advisory database before the weekly audit runs.

- **Weekly security audit** — Added `.github/workflows/weekly-audit.yml`. Runs every Monday
  at 06:00 UTC and can be triggered manually via `workflow_dispatch`:
  - Audits Python deps with `pip-audit` against the `uv`-exported lockfile
  - Audits frontend JS deps with `npm audit`
  - Writes a full report to the workflow step summary
  - If high- or critical-severity vulnerabilities are found, opens (or updates) a GitHub Issue
    labelled `security-audit` + `maintenance` with the full audit output

### Dependency status

**Python backend (`pyproject.toml`):**
All packages use `>=` lower-bound constraints pinned by `uv.lock`. Constraints updated since
the 2026-06-10 cycle to track newer minimum versions.

| Package | Constraint | Status |
|---|---|---|
| `fastapi` | `>=0.136.0` | Current |
| `langchain` | `>=1.3.0` | Current |
| `langgraph` | `>=1.2.0` | Current |
| `chromadb` | `>=1.5.0` | Current |
| `uvicorn[standard]` | `>=0.49.0` | Current |
| `langchain-google-genai` | `>=4.2.0` | Current |
| `langchain-openai` | `>=1.3.0` | Current |
| `langchain-mistralai` | `>=1.1.0` | Current |
| `langchain-ollama` | `>=1.1.0` | Current |
| `langchain-docling` | `>=2.0.0` | Current |
| `redis` | `>=8.0.0` | Current |
| `docling` | `>=2.101.0` | Current |
| `pypdf` | `>=6.0.0` | Current |
| `python-dotenv` | `>=1.2.2` | Current |

**Frontend (`frontend/package.json`):**

| Package | Constraint | Status |
|---|---|---|
| `react` / `react-dom` | `^19.2.5` | Current |
| `react-markdown` | `^10.1.0` | Current |
| `vite` | `^8.0.0` | Current |
| `@vitejs/plugin-react` | `^6.0.0` | Current |
| `typescript` | `~6.0.0` | Current |
| `eslint` | `^10.0.0` | Current |
| `vitest` | `^4.1.7` | Current |
| `@playwright/test` | `^1.60.0` | Current |

### No major upgrades pending
All packages are on current major versions this cycle. Dependabot will open PRs for any
future updates automatically.

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
