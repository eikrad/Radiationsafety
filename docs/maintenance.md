# Maintenance Log

Weekly dependency and health checks for the Radiationsafety RAG project.

---

## 2026-07-01

### Checks performed
- Reviewed all dependencies in `pyproject.toml` / `uv.lock` and `frontend/package.json` / `package-lock.json`
- Ran `uvx pip-audit` against the synced `uv` environment for known CVEs/GHSAs
- Ran `npm -C frontend outdated` and `npm -C frontend audit` for outdated/vulnerable frontend packages
- Reviewed `.github/workflows/ci.yml` and `.github/workflows/protect-master.yml`
- Checked CI run history for `staging` via the GitHub Actions API (last run: 2026-06-17, `success`)

### Fixes applied

- **Security (Python)** — `uv lock --upgrade-package` for the packages below, fixing 5 known vulnerabilities reported by `pip-audit`:
  - `langchain` `1.3.7` → `1.3.11` (fixes `GHSA-gr75-jv2w-4656`, path-confinement issue in filesystem-resolving components; fix shipped in `1.3.9`)
  - `langsmith` `0.8.14` → `0.9.5` (fixes `GHSA-f4xh-w4cj-qxq8`, arbitrary file read via `TracingMiddleware`; fix shipped in `0.8.18`)
  - `pydantic-settings` `2.14.1` → `2.14.2` (fixes `GHSA-4xgf-cpjx-pc3j`, symlink traversal in `NestedSecretsSettingsSource`)
  - `pypdf` `6.13.2` → `6.14.2` (fixes `GHSA-jm82-fx9c-mx94`, memory-exhaustion DoS when `/Length` is absent from a content stream; bumped the `pyproject.toml` floor to `>=6.13.3`)
  - `starlette` `1.3.0` → `1.3.1` (fixes `PYSEC-2026-249`, `max_fields`/`max_part_size` limits silently ignored for some form parsing paths)
  - Followed by a full `uv lock --upgrade` to also pick up in-range patch/minor bumps for ~50 transitive packages (fastapi, langchain-openai, docling stack, pytest, redis, numpy, torch, etc.) — see dependency table below for the direct-dependency deltas.
- **Security (frontend)** — `npm -C frontend audit fix` (no `--force`), resolving 4 vulnerabilities (1 low, 3 high) in dev-only transitive packages (`@babel/core` arbitrary file read, `flatted` DoS/prototype pollution, `undici` WebSocket/request-smuggling issues, `vite` `server.fs.deny` bypass). Followed by `npm -C frontend update` to bring all in-range packages to their "wanted" latest versions (react, vite, eslint, vitest, etc.). `npm audit` now reports 0 vulnerabilities. No entries in `package.json` itself needed to change (all caret ranges already covered the resolved versions).
- **CI config bug** — `.github/workflows/ci.yml` `docker-integration` job diffed against `origin/main...HEAD` to detect Docker file changes, but this repo has no `main` branch (only `master`/`staging`), so the diff always failed silently and the Docker-changed check likely never triggered correctly on non-`master`-based comparisons. Fixed to `origin/master...HEAD`.
- Removed a stray `json` file accidentally created in the repo root by an earlier `pip-audit` invocation during this session (not part of the codebase, deleted before commit).

### Vulnerabilities reviewed but not fixed (no upstream patch)

| Package | Version | Advisory | Why not fixed | Risk in this codebase |
|---|---|---|---|---|
| `chromadb` | `1.5.9` (latest) | `PYSEC-2026-311` / `CVE-2026-45829` ("ChromaToast") | Pre-auth code injection via `trust_remote_code=true` on the Chroma **server's** `/api/v2/.../collections` endpoint when resolving a Hugging Face embedding model. Unpatched upstream as of `1.5.9`. | Not exploitable here: this project uses `chromadb.PersistentClient` (embedded, in-process — see `ingestion.py`), never `HttpClient`/server mode, and embeddings are always Gemini per `AGENTS.md` (`trust_remote_code` is never set). No Chroma server is exposed. Documented here for awareness; re-check when chroma-core ships a fix. |
| `torch` | was `2.12.0`, now `2.12.1` (transitive, via `docling`) | `CVE-2025-3000` | Local-attack-vector (`AV:L`) memory corruption in `torch.jit.script`. No upstream fix version was listed by `pip-audit` at the time of the `2.12.0` scan. | The routine `uv lock --upgrade` pass happened to move `torch` `2.12.0` → `2.12.1` as an in-range transitive resolution, which resolved the finding (no longer reported by `pip-audit` after re-scan). Codebase never calls `torch.jit.script` on untrusted input (docling uses pretrained models internally), so exposure was already low regardless. |

### Dependency status

**Python backend (`pyproject.toml`) — direct dependencies:**

| Package | Old constraint | New constraint | Status |
|---|---|---|---|
| `pypdf` | `>=6.0.0` | `>=6.13.3` | Bumped (security fix, see above) |
| `chromadb` | `>=1.5.0` | `>=1.5.0` (resolves to `1.5.9`) | Current; known unpatched CVE noted above |
| `fastapi` | `>=0.136.0` | `>=0.136.0` (resolves to `0.139.0`) | Current |
| `langchain` | `>=1.3.0` | `>=1.3.0` (resolves to `1.3.11`) | Current |
| `langgraph` | `>=1.2.0` | `>=1.2.0` (resolves to `1.2.7`) | Current |
| `langchain-openai` | `>=1.3.0` | `>=1.3.0` (resolves to `1.3.3`) | Current |
| `langchain-google-genai` | `>=4.2.0` | `>=4.2.0` (resolves to `4.2.6`) | Current |
| `langchain-mistralai` | `>=1.1.0` | `>=1.1.0` | Current |
| `redis` | `>=8.0.0` | `>=8.0.0` (resolves to `8.0.1`) | Current |
| `docling` | `>=2.101.0` | `>=2.101.0` (resolves to `2.108.0`) | Current |
| `uvicorn[standard]` | `>=0.49.0` | `>=0.49.0` | Current |

**Frontend (`frontend/package.json`):**
All major deps remain on the same major versions; in-range patch/minor versions refreshed via `npm update` (react `19.2.5`→`19.2.7`, vite `8.0.14`→`8.1.2`, eslint `10.4.0`→`10.6.0`, vitest `4.1.7`→`4.1.9`, `@typescript-eslint/*` `8.60.0`→`8.62.1`, `@playwright/test` `1.60.0`→`1.61.1`, others similarly refreshed).

| Package | Constraint | Status |
|---|---|---|
| `react` / `react-dom` | `^19.2.5` | Current |
| `react-markdown` | `^10.1.0` | Current |
| `vite` | `^8.0.0` | Current |
| `typescript` | `~6.0.0` | Current |
| `eslint` | `^10.0.0` | Current |
| `vitest` | `^4.1.7` | Current |
| `@vitejs/plugin-react` | `^6.0.0` | Current |

### Major upgrades pending

Not applied this cycle — flagged for manual review since they cross a major/breaking version boundary:

| Package | Current | Latest | Notes |
|---|---|---|---|
| `jsdom` (frontend, dev) | `28.1.0` | `29.1.1` | Major bump; used only by Vitest's DOM environment — low risk but should be tested in isolation before adopting |
| `globals` (frontend, dev) | `16.5.0` | `17.7.0` | Major bump; ESLint flat-config globals list, check for renamed/removed global sets |
| `eslint-plugin-react-refresh` (frontend, dev) | `0.4.26` | `0.5.3` | Pre-1.0 package, treat as breaking; check rule-set changes before bumping |
| `protobuf` (Python, transitive via `google-genai`/opentelemetry) | `6.33.6` | `7.35.1` | Major bump; pulled in transitively, not pinned directly — left alone this cycle since `uv lock --upgrade` did not select it (stayed on `6.x` within current constraints) and a `7.x` protobuf runtime change warrants explicit testing of the Gemini/OTel code paths first |

### Notes

- All changes verified locally before commit: `uv run pytest tests/ -v` (177 passed), `uv run ruff check .` / `black --check .` / `isort --check .` (all clean), `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip` (clean), `npm -C frontend run lint` (clean), `npm -C frontend run build` (success), `npm -C frontend run test` (29 passed).
- `uv run pre-commit run --all-files` could not complete in this sandbox (outbound git fetch to `github.com/psf/black` blocked by the proxy, HTTP 403) — not a repo issue; the equivalent checks (`black --check`, `isort --check`) were run directly instead and passed.
- Chroma collection names (`radiation-iaea`, `radiation-dk-law`) were not touched.
- No `.env.example` changes needed — no new environment variables introduced.

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
