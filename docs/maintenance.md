# Maintenance Log

Weekly dependency and health checks for the Radiationsafety RAG project.

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
