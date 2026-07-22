# Maintenance Log

Weekly dependency and health checks for the Radiationsafety RAG project.

---

## 2026-07-22

> **⚠️ Open-PR backlog — action needed.** Five PRs from prior maintenance cycles are still open and unmerged against `staging`: **#56** (staging→master release merge), **#57** (2026-07-08 maintenance), **#58** (docs), **#59** (2026-07-15 maintenance), **#60** (docs). Because none of these has landed, `staging`'s tip going into this cycle was still the 2026-06-17 entry below — none of the 06-24, 07-01, 07-08, or 07-15 cycles' changes are actually present on `staging`. This audit was performed against the real, current `origin/staging` tip (not assumed-merged prior PRs). Every additional unmerged maintenance PR compounds the backlog and risks conflicting/duplicate dependency bumps landing out of order. **Recommendation: review and merge (or explicitly close) #56–#60 before the next cycle** so future audits start from a moving baseline instead of stacking on top of each other.

### Checks performed
- Fetched `origin/staging` and reset the working branch to its actual current tip (verified, not assumed)
- Baseline run **before any changes**: `uv sync --all-extras && uv run pytest tests/ -v` → **177 passed**
- Baseline run: `npm -C frontend ci && npm -C frontend run test && npm -C frontend run lint && npm -C frontend run build` → **29 tests passed (5 files)**, lint clean, build clean
- Also ran the full CI lint/type set locally (`ruff check`, `black --check`, `isort --check`, `mypy`, `pre-commit run --all-files`) — all clean on baseline, no pre-existing issues found
- Python dependency audit: `uv pip list --outdated` reviewed against `pyproject.toml` floors
- Frontend dependency audit: `npm -C frontend outdated`
- Security audit: `npm -C frontend audit` and `uv run --with pip-audit pip-audit -r <exported uv.lock requirements>` (project interpreter, not global)

### Fixes applied
No pre-existing failures found — baseline was fully green, nothing to fix. Applied targeted dependency bumps only (see below).

### Dependency updates

**Python backend** — applied via `uv lock --upgrade-package <name>` one at a time (never a blanket `uv lock --upgrade`), all within `pyproject.toml`'s existing `>=` floors:

| Package | Before | After | Type | Notes |
|---|---|---|---|---|
| `fastapi` | 0.139.0 | 0.139.2 | patch | |
| `langchain` | 1.3.11 | 1.3.14 | patch | |
| `langchain-core` | 1.4.8 | 1.5.0 | minor | pulled transitively by langchain bump |
| `langchain-google-genai` | 4.2.6 | 4.3.1 | minor | |
| `langchain-mistralai` | 1.1.5 | 1.1.6 | patch | |
| `langchain-openai` | 1.3.3 | 1.4.0 | minor | pulled `openai` 2.44.0 → 2.47.0 transitively |
| `langgraph` | 1.2.7 | 1.2.9 | patch | |
| `uvicorn` | 0.49.0 | 0.51.0 | minor | |
| `mypy` (dev) | 2.1.0 | 2.3.0 | minor | |
| `pre-commit` (dev) | 4.6.0 | 4.6.1 | patch | |
| `ruff` (dev) | 0.15.20 | 0.15.22 | patch | |
| `pyasn1` | 0.6.3 | 0.6.4 | patch | fixes 3 CVEs, see Security findings |
| `setuptools` | 81.0.0 | 83.0.0 | minor | fixes 1 CVE, see Security findings |
| `torch` | 2.12.1 | 2.13.0 | minor | fixes 1 CVE, see Security findings |
| `torchvision` | 0.27.1 | 0.28.0 | minor | pulled transitively by torch bump |

`docling` (2.108.0 → 2.114.0 available) was **deliberately left untouched this cycle** — it and `rapidocr` sit upstream of `opencv-python`, which currently has a major-version jump available (4.13.0.92 → 5.0.0.93). A `docling` bump was not attempted this cycle to avoid risking that transitive major landing unreviewed; confirmed via lockfile diff that `opencv-python` stayed at `4.13.0.92` throughout.

**Frontend** — applied via `npm -C frontend update` (package.json semver ranges unchanged, `package-lock.json` refreshed only):

| Package | Before | After | Type |
|---|---|---|---|
| `@typescript-eslint/eslint-plugin` | 8.62.1 | 8.65.0 | minor |
| `@typescript-eslint/parser` | 8.62.1 | 8.65.0 | minor |
| `@vitejs/plugin-react` | 6.0.3 | 6.0.4 | patch |
| `eslint` | 10.6.0 | 10.7.0 | minor |
| `react` | 19.2.7 | 19.2.8 | patch |
| `react-dom` | 19.2.7 | 19.2.8 | patch |
| `vite` | 8.1.2 | 8.1.5 | patch |
| `vitest` | 4.1.9 | 4.1.10 | patch |

### Security findings

`npm -C frontend audit`: **0 vulnerabilities.**

`pip-audit` against the exported `uv.lock` (via `uv run --with pip-audit pip-audit -r <export>`, project interpreter):

Before fixes — 7 known vulnerabilities in 4 packages:

| Package | Version | ID | Severity/notes | Resolution |
|---|---|---|---|---|
| `torch` | 2.12.1 | PYSEC-2025-194 | Memory corruption in `torch.jit.script`, publicly disclosed exploit, requires local access | **Fixed** — upgraded to 2.13.0 |
| `setuptools` | 81.0.0 | PYSEC-2026-3447 | Unicode NFC/NFD filename-normalization bypass of `MANIFEST.in` excludes during sdist build (macOS APFS/HFS+ only) | **Fixed** — upgraded to 83.0.0 |
| `pyasn1` | 0.6.3 | PYSEC-2026-3455 / -3456 / -3457 | 3× algorithmic-complexity DoS in BER/CER/DER decoding of untrusted ASN.1 (long-form tags, OID arcs, `REAL` exponent) | **Fixed** — upgraded to 0.6.4 |
| `chromadb` | 1.5.9 | PYSEC-2026-311 | Pre-auth code injection via `trust_remote_code` on the Chroma HTTP server's `/api/v2/.../collections` endpoint | **Not directly fixable** — no fix version published yet (1.5.9 is the latest release on PyPI as of this cycle; confirmed via `pip index versions`) |

**Risk assessment — chromadb PYSEC-2026-311 (remaining, unpatched upstream):** The vulnerable surface is the standalone Chroma HTTP/gRPC server accepting a `trust_remote_code` flag on collection creation. This app never runs that server component — `ingestion.py` only ever instantiates `chromadb.PersistentClient` / `langchain_community.vectorstores.Chroma` embedded in-process against a local on-disk directory (`_CHROMA_DIR`); no Chroma server process is started and no such HTTP endpoint is exposed by this application. **Assessed as not reachable in this app's current usage.** Recommend re-checking each cycle until a fix version ships, in case usage patterns change (e.g. a future move to a networked Chroma server).

### Major upgrades — flagged, NOT applied

| Package | Current | Available | Why held back |
|---|---|---|---|
| `docling` (Python) | 2.108.0 | 2.114.0 | Minor version itself, but sits upstream of `opencv-python`'s pending major (below) via `rapidocr`; held back this cycle out of caution rather than bump-and-hope — needs a dedicated review of the OCR dependency chain before touching |
| `opencv-python` (Python, transitive via docling/rapidocr) | 4.13.0.92 | 5.0.0.93 | Major version bump; known breaking-change risk for docling's OCR path — do not pull in via a blanket `uv lock --upgrade` |
| `@testing-library/jest-dom` (frontend) | 6.9.1 | 7.0.0 | Major; requires reviewing breaking changes to matcher API before adopting |
| `globals` (frontend) | 16.5.0 | 17.7.0 | Major; flagged in the 2026-06-17 cycle too, still not blocking (ESLint 10 supports both) — revisit if it keeps recurring |
| `jsdom` (frontend) | 28.1.0 | 29.1.1 | Major; test-environment dependency, needs a dedicated compatibility check with vitest 4.x before bumping |
| `eslint-plugin-react-refresh` (frontend) | 0.4.26 | 0.5.3 | 0.x → 0.x "minor" that is breaking-change-equivalent per semver-zero convention; caret range (`^0.4.24`) correctly excludes it from `npm update` |
| `typescript` (frontend) | 6.0.3 | 7.0.2 | Major; package.json pins `~6.0.0` deliberately — needs its own upgrade cycle given TS 7's compiler rewrite |

### Post-change verification
All checks re-run after dependency bumps, everything green:
- `uv run pytest tests/ -v` → **177 passed**
- `uv run ruff check .` → clean
- `uv run black --check .` → clean
- `uv run isort --check .` → clean
- `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip` → clean
- `uv run pre-commit run --all-files` → clean
- `npm -C frontend run test` → **29 passed (5 files)**
- `npm -C frontend run lint` → clean
- `npm -C frontend run build` → clean
- Confirmed via `git diff --stat` that only `uv.lock` and `frontend/package-lock.json` changed — no `pyproject.toml` or `package.json` range edits were needed

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
