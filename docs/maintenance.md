# Maintenance Log

Weekly dependency and health checks for the Radiationsafety RAG project.

---

## 2026-08-19

> **Note on `docs/maintenance.md` staleness:** this file's history lives on `staging`, and `staging` currently has entries through 2026-08-12 — this cycle is not actually two months late. The "last entry 2026-06-17" observation applies only to `master`, which is **61 commits behind `staging`** (`git rev-list --left-right --count origin/master...origin/staging` → `1  61`) — confirmed by `git show origin/master:docs/maintenance.md`, whose latest entry is indeed the 2026-06-17 one. Every weekly cycle since 2026-07-08 has landed correctly on `staging` (see git log: `54f187c`, `b9dab7b`, `9250bb8`, `b001c83`, `f2cf6be`); they just haven't been promoted to `master` via a `staging`→`master` merge. Recommend the owner merge `staging` → `master` to bring the production branch's log (and code) current — a ~2-month-old `master` means production is missing 5 cycles of dependency/security fixes. Open-PR backlog verified fresh via `mcp__github__list_pull_requests` (state=open): **0 open PRs**, matching the pre-verified context for this cycle.

### Checks performed
- `git fetch origin staging master`, then `git reset --hard origin/staging` (branch `claude/modest-faraday-jdicez`, tip `4d370de`) — starting fresh per this cycle's setup, no prior local commits to reconcile.
- Verified open-PR backlog live via `mcp__github__list_pull_requests` (state=open) → 0, confirming the pre-checked context.
- Baseline **before any changes**: `uv sync --all-extras && uv run pytest tests/ -n auto -v` → **177 passed**; `uv run ruff check .`, `uv run black --check .`, `uv run isort --check .`, `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip`, `uv run pre-commit run --all-files` → all clean.
- Baseline: `npm -C frontend ci && npm -C frontend run test` → **29 passed (5 files)**; `npm -C frontend run lint` and `npm -C frontend run build` → clean.
- Playwright E2E (`npx playwright install --with-deps chromium && npm run test:e2e`, as run in CI) could **not** be executed in this sandbox — the outbound proxy blocks `cdn.playwright.dev` (`403 request blocked: no rule or allowlist entry allows host`). This is a sandbox network restriction, not a code regression; not run in any prior maintenance cycle either (same environment).
- Python dependency audit: `uv pip list --outdated` cross-checked against prior cycle's transitive-pinning notes (opencv-python/antlr4-python3-runtime still pinned via docling/rapidocr/omegaconf; nvidia-*/cuda-* packages still pinned by torch's CUDA runtime requirement — none moved).
- Frontend dependency audit: `npm -C frontend outdated`.
- Security audit: `uv export --format requirements-txt --no-hashes` + `uv run --with pip-audit pip-audit -r <export>` (project's own Python 3.12 interpreter — a bare `pip-audit -r <export>` against the live index still fails resolving point releases newer than published upstream, same issue as every prior cycle); `npm -C frontend audit`.
- Verified `langchain-openai` 1.4.3→1.5.2 does **not** pull the `openai` 3.x major transitively (checked `uv.lock` before/after in isolation — `openai` stayed pinned at `2.53.0`).

### Fixes applied
No pre-existing failures found — baseline was fully green (177 pytest, all lint/format/type checks clean, frontend 29 tests/lint/build clean). No code fixes needed. Applied a batch of routine patch/minor dependency bumps only (no new security vulnerabilities found this cycle).

### Security findings

**`pip-audit`** — **2 known vulnerabilities in 2 packages**, both carried over unchanged from 2026-08-12, no new findings:

| Package | Version | ID | Resolution |
|---|---|---|---|
| `chromadb` | 1.5.9 | PYSEC-2026-311 | **Not fixable** — still latest on PyPI; **not reachable** (embedded `PersistentClient` only, no HTTP server, `trust_remote_code` never set) |
| `cryptography` | 49.0.0 | PYSEC-2026-3552 | **Fix requires major bump (50.0.0)** — not applied, see Major upgrades below; **not reachable** (transitive-only via `google-auth`, no PKCS#7/S-MIME usage) |

**`npm -C frontend audit`** — **0 vulnerabilities**, both before and after changes.

### Dependency updates

**Python backend** — applied via one batched `uv lock --upgrade-package <name> ...` call (each package named explicitly, never a blanket `uv lock --upgrade`), all within `pyproject.toml`'s existing `>=` floors — **no `pyproject.toml` edits needed**:

| Package | Before | After | Type | Notes |
|---|---|---|---|---|
| `langchain-core` | 1.5.4 | 1.6.0 | minor | |
| `langchain-openai` | 1.4.3 | 1.5.2 | minor | verified `openai` stayed pinned at 2.53.0, did not pull the 3.x major |
| `langchain-google-genai` | 4.3.3 | 4.3.4 | patch | |
| `langgraph-sdk` | 0.4.2 | 0.4.3 | patch | |
| `langsmith` | 0.10.18 | 0.11.1 | minor | |
| `docling` | 2.119.0 | 2.120.3 | patch | confirmed `opencv-python` stayed at `4.13.0.92` — same trap avoided as every prior cycle |
| `docling-core` | 2.91.0 | 2.92.0 | patch | |
| `docling-parse` | 7.12.1 | 7.14.0 | minor | |
| `docling-slim` | 2.119.0 | 2.120.3 | patch | |
| `google-genai` | 2.17.0 | 2.18.1 | minor | |
| `huggingface-hub` | 1.27.0 | 1.28.0 | minor | |
| `onnxruntime` | 1.28.0 | 1.29.0 | minor | |
| `pypdf` | 6.15.0 | 6.16.1 | minor | |
| `pypdfium2` | 5.12.1 | 5.13.0 | minor | |
| `tiktoken` | 0.13.0 | 0.14.0 | minor | |
| `uvicorn` | 0.52.1 | 0.52.4 | patch | |
| `orjson` | 3.11.9 | 3.12.0 | minor | |
| `lxml` | 6.1.1 | 6.1.2 | patch | |
| `filelock` | 3.32.2 | 3.32.3 | patch | |
| `idna` | 3.18 | 3.19 | minor | |
| `librt` | 0.13.0 | 0.15.0 | minor | |
| `mail-parser` | 4.6.1 | 4.6.2 | patch | |
| `platformdirs` | 4.11.2 | 4.11.3 | patch | |
| `pygments` | 2.20.0 | 2.21.0 | minor | |
| `python-dotenv` | 1.2.2 | 1.2.3 | patch | |
| `charset-normalizer` | 3.5.0 | 3.5.1 | patch | |
| `ruff` (dev) | 0.16.2 | 0.16.3 | patch | |
| `mypy` (dev) | 2.3.0 | 2.3.1 | patch | |
| `typer`, `mpmath`, `packaging`, `pydantic-core`, `tokenizers` | — | — | — | requested but did not move — held at current versions by other packages' compatibility constraints elsewhere in the tree |

**Frontend** — applied via `npm -C frontend update` (package.json semver ranges unchanged, `package-lock.json` refreshed only):

| Package | Before | After | Type |
|---|---|---|---|
| `@testing-library/user-event` | 14.6.4 | 14.6.5 | patch |
| `@typescript-eslint/eslint-plugin` | 8.66.0 | 8.67.0 | patch |
| `@typescript-eslint/parser` | 8.66.0 | 8.67.0 | patch |
| `vitest` | 4.1.10 | 4.1.11 | patch |
| assorted transitive patch bumps | — | — | — |

### Major upgrades — flagged, NOT applied

| Package | Current | Available | Why held back |
|---|---|---|---|
| `cryptography` (Python, transitive via `google-auth`) | 49.0.0 | 50.0.0 | Major; fixes PYSEC-2026-3552 but not reachable in this app — carried over, still needs dedicated review |
| `opencv-python` (Python, transitive via `docling`/`rapidocr`) | 4.13.0.92 | 5.0.0.93 | Major; known breaking-change risk for docling's OCR path — held back every cycle since 2026-07-08 |
| `semchunk` (Python, transitive via `docling`) | 3.2.5 | 4.1.1 | Major; carried over from 2026-07-15 |
| `websockets` (Python, transitive) | 15.0.1 | 17.0.1 | Major (two majors behind); carried over |
| `transformers` (Python, transitive via `docling`/`torch`) | 5.8.1 | 5.15.1 | Large jump tied to `docling`/`torch` compatibility — deferred, gap has grown further this cycle |
| `openai` (Python, transitive via `langchain-openai`) | 2.53.0 | 3.3.1 | Major SDK rewrite; confirmed `langchain-openai` 1.5.2 still pins below 3.x — gap has grown since 2026-08-12 (was 3.0.0, now 3.3.1) |
| `langchain-docling` (Python) | 2.0.0 | 3.0.0 | **New this cycle** — major; not bumped, needs its own compatibility review against `docling` 2.x |
| `xxhash` (Python, transitive via `langgraph`/`langsmith`) | 3.8.1 | 4.0.1 | Major; no CVE driving it, carried over |
| `antlr4-python3-runtime` (Python, transitive via `omegaconf`←`rapidocr`←`docling`) | 4.9.3 | 4.13.2 | Large jump, transitive-pinned by `omegaconf`'s compatibility range — same category as `opencv-python` |
| `ast-serialize` (Python, transitive via `mypy`, dev-only) | 0.6.0 | 0.8.0 | 0.x "zero-ver" jump, breaking-change-equivalent per convention — dev-only, low risk, held back |
| `eslint-plugin-react-refresh` (frontend) | 0.4.26 | 0.5.4 | 0.x "minor" that is breaking-change-equivalent per semver-zero convention — no open Dependabot PR tracking it currently |
| `typescript` (frontend) | 6.0.3 | 7.0.2 | Major (TS7 "Corsa" native compiler rewrite) — no open Dependabot PR tracking it currently |

### Infrastructure pins reviewed — no drift
- `Dockerfile`: `python:3.12-slim` — matches CI's `python-version: "3.12"` and `.python-version`
- `frontend/Dockerfile`: `node:22-alpine` — matches CI's `node-version: "22"`
- `nginx:alpine` (frontend/Dockerfile stage 2) — floating tag, no pinned-stale patch to bump

### Post-change verification
All checks re-run after dependency bumps, everything green:
- `uv run pytest tests/ -n auto -v` → **177 passed**
- `uv run ruff check .` → clean
- `uv run black --check .` → clean
- `uv run isort --check .` → clean
- `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip` → clean
- `uv run pre-commit run --all-files` → clean (black, isort)
- `npm -C frontend run test` → **29 passed (5 files)**
- `npm -C frontend run lint` → clean
- `npm -C frontend run build` → clean
- `npm -C frontend audit` → 0 vulnerabilities
- `pip-audit` → same 2 known, non-reachable findings as before changes (no new vulnerabilities)
- Confirmed via `git diff --stat` that only `uv.lock` and `frontend/package-lock.json` changed — no `pyproject.toml` or `package.json` range edits were needed

---

## 2026-08-12

> **Open-PR backlog — verified fresh via `mcp__github__list_pull_requests` (state=open).** Only **2** PRs open against `staging`, both Dependabot: **#80** (`globals` 17.7.0→17.9.0) and **#79** (`vite` 8.2.0→8.2.1). This is a markedly healthier state than the 2026-07-22/2026-08-05 cycles, which each flagged a growing backlog of unmerged prior maintenance PRs (#56–#60, #74, #71, #75) — that backlog appears to have been cleared since the last cycle. Both open packages were explicitly excluded from this cycle's dependency-update pass to avoid duplicate/conflicting bumps.

### Checks performed
- `git fetch origin`, then diffed the local branch (`claude/modest-faraday-l94b2x`) both directions against `origin/staging`. The only "unique" local commit (`39182e8`, a merge-PR-#56-into-master wrapper) had **zero diff** vs. the merge-base — confirmed via `git diff <merge-base> 39182e8 --stat` (empty) — so it carried no real content. Reset the branch to `origin/staging`'s actual tip (`8082b30`) rather than risk building on a stale/diverged seed, per the note in prior cycles that this branch sometimes gets seeded from an older point.
- Verified the open-PR backlog live via GitHub MCP tools (see banner above), not from any carried-over list.
- Baseline **before any changes**: `uv sync --all-extras && uv run pytest tests/ -v` → **177 passed**; `uv run ruff check .`, `uv run black --check .`, `uv run isort --check .`, `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip`, `uv run pre-commit run --all-files` → all clean.
- Baseline: `npm -C frontend ci && npm -C frontend run test` → **29 passed (5 files)**; `npm -C frontend run lint` and `npm -C frontend run build` → clean.
- Python dependency audit: `uv pip list --outdated` cross-checked with `uv tree --invert` to confirm which outdated packages are direct vs. transitive-only, and which transitive majors (e.g. `opencv-python`, `antlr4-python3-runtime`) are pinned by a parent package's compatibility range (`docling`/`rapidocr`/`omegaconf`) rather than by our own constraints.
- Frontend dependency audit: `npm -C frontend audit`, `npm -C frontend outdated`, excluding the 2 packages already covered by open Dependabot PRs (`globals`, `vite`).
- Security audit: `uv export --format requirements-txt --no-hashes` + `uv run --with pip-audit pip-audit -r <export> --desc` (project's own Python 3.12 interpreter, not a fresh resolve against the live index — a bare `pip-audit -r <export>` still fails trying to resolve point releases newer than what's published upstream, same issue noted in every prior cycle); `npm -C frontend audit`.
- Re-verified the chromadb `PYSEC-2026-311` finding against current `ingestion.py`: grepped for `Chroma`/`HttpClient`/`PersistentClient`/`trust_remote_code` — still only `chromadb.PersistentClient` (embedded, on-disk) at `ingestion.py:123` and `:760`, no server component, no `trust_remote_code`. Confirmed via `pip index versions chromadb` that `1.5.9` is still the latest release — no fix exists upstream yet.
- Re-verified the cryptography `PYSEC-2026-3552` finding: grepped the codebase for `pkcs7`/`cryptography` usage directly — none found; `cryptography` is transitive-only via `google-auth`, which uses it for JWT/service-account signing, not S/MIME `EnvelopedData` decryption. Confirmed `50.0.0` (the fix) is available via `pip index versions cryptography` but not applied — see Major upgrades below.

### Fixes applied
No pre-existing failures found — baseline was fully green (177 pytest, all lint/format/type checks clean, frontend 29 tests/lint/build clean). Applied one real security fix (`pypdf`), one transitive security fix (`nanoid` via `npm audit fix`), and a batch of routine patch/minor dependency bumps.

### Security findings

**`pip-audit` (project interpreter, `--desc`)** — **4 known vulnerabilities in 3 packages** found before fixes:

| Package | Version | ID | Severity/notes | Resolution |
|---|---|---|---|---|
| `pypdf` | 6.14.2 | PYSEC-2026-3655 | Crafted PDF with an oversized `/ToUnicode` font entry causes large memory consumption during text extraction | **Fixed** — upgraded to 6.15.0 |
| `pypdf` | 6.14.2 | PYSEC-2026-3656 | Crafted PDF with oversized font-width entries causes long runtimes / large memory consumption during text extraction | **Fixed** — upgraded to 6.15.0 |
| `chromadb` | 1.5.9 | PYSEC-2026-311 | Pre-auth code injection via `trust_remote_code` on the Chroma HTTP server's collections endpoint | **Not fixable** — 1.5.9 still latest on PyPI; **not reachable** in this app (embedded `PersistentClient` only, no HTTP server, `trust_remote_code` never set) |
| `cryptography` | 49.0.0 | PYSEC-2026-3552 | Bleichenbacher-style oracle in `pkcs7_decrypt_der`/`_pem`/`_smime` against attacker-supplied S/MIME `EnvelopedData` | **Fix requires major bump (50.0.0)** — not applied, see Major upgrades below; **not reachable** (transitive-only via `google-auth`, no PKCS#7/S-MIME usage in this codebase) |

`pypdf` is a **new, directly relevant** finding this cycle — both CVEs are parser-triggered DoS conditions reachable through this app's own PDF ingestion path (`ingestion.py`, `documents/IAEA/`), so this was treated as a priority fix, not just a routine bump.

**`npm -C frontend audit`** — **1 high-severity vulnerability** found before fixes:

| Package | Version | Severity | Advisory | Resolution |
|---|---|---|---|---|
| `nanoid` | 3.3.16 | High | GHSA-2v37-7h3g-55p8 — custom ID generators can loop indefinitely when size is 0 | **Fixed** — 3.3.18, via `npm audit fix` (transitive via `postcss`←`vite`; only `nanoid`'s lockfile entry changed, no `package.json` range edits, no platform-optional packages actually installed) |

After fixes: `npm -C frontend audit` → **0 vulnerabilities**.

### Dependency updates

**Python backend** — applied via one batched `uv lock --upgrade-package <name> ...` call (each package named explicitly, never a blanket `uv lock --upgrade`), all within `pyproject.toml`'s existing `>=` floors — **no `pyproject.toml` edits needed**:

| Package | Before | After | Type | Notes |
|---|---|---|---|---|
| `pypdf` | 6.14.2 | 6.15.0 | minor | **security fix**, see above |
| `langchain` | 1.3.14 | 1.3.15 | patch | |
| `langchain-core` | 1.5.3 | 1.5.4 | patch | |
| `langchain-google-genai` | 4.3.2 | 4.3.3 | patch | |
| `langchain-openai` | 1.4.1 | 1.4.3 | patch | |
| `langgraph` | 1.2.10 | 1.2.11 | patch | |
| `langgraph-checkpoint` | 4.1.1 | 4.2.0 | minor | |
| `langsmith` | 0.10.16 | 0.10.18 | patch | |
| `docling` | 2.118.0 | 2.119.0 | patch | confirmed `opencv-python` stayed at `4.13.0.92` — same trap avoided as every prior cycle |
| `docling-core` | 2.90.0 | 2.91.0 | patch | pulled transitively by docling bump |
| `docling-ibm-models` | 3.13.3 | 3.14.0 | minor | |
| `docling-parse` | 7.10.0 | 7.12.1 | minor | |
| `docling-slim` | 2.118.0 | 2.119.0 | patch | pulled in two new small transitive deps, `olefile` 0.47 and `python-oxmsg` 0.0.2 (Outlook `.msg` parsing support) — reviewed, both small/uncontroversial |
| `google-auth` | 2.56.2 | 2.56.3 | patch | |
| `google-genai` | 2.16.0 | 2.17.0 | minor | |
| `googleapis-common-protos` | 1.75.0 | 1.75.1 | patch | |
| `greenlet` | 3.5.4 | 3.5.5 | patch | |
| `huggingface-hub` | 1.26.0 | 1.27.0 | minor | |
| `mail-parser` | 4.5.0 | 4.6.1 | minor | |
| `marko` | 2.2.3 | 2.2.4 | patch | |
| `numpy` | 2.5.1 | 2.5.2 | patch | |
| `platformdirs` | 4.11.0 | 4.11.2 | patch | |
| `pybase64` | 1.4.3 | 1.5.0 | minor | |
| `pydantic-settings` | 2.14.2 | 2.15.0 | minor | |
| `python-discovery` | 1.5.1 | 1.5.2 | patch | |
| `setuptools` | 83.0.0 | 84.0.0 | minor | |
| `soupsieve` | 2.9.1 | 2.9.2 | patch | |
| `sqlalchemy` | 2.0.51 | 2.0.52 | patch | |
| `starlette` | 1.4.1 | 1.6.0 | minor (2 releases) | transitive via `fastapi`, within its allowed range |
| `typing-inspection` | 0.4.2 | 0.4.4 | patch | |
| `charset-normalizer` | 3.4.9 | 3.5.0 | minor | |
| `ruff` (dev) | 0.16.1 | 0.16.2 | patch | |
| `pre-commit` (dev) | 4.6.1 | 4.6.2 | patch | |
| `virtualenv` (dev) | 21.7.1 | 21.7.4 | patch | |
| `mpmath`, `packaging`, `pydantic-core`, `tokenizers`, `typer` | — | — | — | requested but did not move — held at current versions by other packages' compatibility constraints elsewhere in the tree; not a problem, just noting they weren't silently skipped |

**Frontend** — no `package.json` range bumps this cycle (the two outdated-but-safe direct deps, `globals` and `vite`, are both already covered by open Dependabot PRs #80/#79 and were deliberately left alone to avoid a conflicting bump). Only the `nanoid` security fix above touched `frontend/package-lock.json`.

### Major upgrades — flagged, NOT applied

| Package | Current | Available | Why held back |
|---|---|---|---|
| `cryptography` (Python, transitive via `google-auth`) | 49.0.0 | 50.0.0 | Major; fixes PYSEC-2026-3552 but assessed not reachable in this app (see Security findings) — carried over from 2026-08-05, still needs a dedicated review rather than a drive-by major bump |
| `opencv-python` (Python, transitive via `docling`/`rapidocr`) | 4.13.0.92 | 5.0.0.93 | Major; known breaking-change risk for docling's OCR path — held back every cycle since 2026-07-08 |
| `semchunk` (Python, transitive via `docling`) | 3.2.5 | 4.1.1 | Major; carried over from 2026-07-15 |
| `websockets` (Python, transitive) | 15.0.1 | 17.0.1 | Major (two majors behind); carried over, no security relevance found this cycle |
| `transformers` (Python, transitive via `docling`/`torch`) | 5.8.1 | 5.15.0 | Large jump tied to `docling`/`torch` compatibility — deferred pending dedicated review, carried over from 2026-07-15, gap has grown further this cycle |
| `openai` (Python, transitive via `langchain-openai`) | 2.53.0 | 3.0.0 | **New this cycle** — major SDK rewrite; `langchain-openai` 1.4.3's pin did not pull it in, so left alone pending confirmation `langchain-openai` supports it |
| `xxhash` (Python, transitive via `langgraph`/`langsmith`) | 3.8.1 | 4.0.0 | **New this cycle** — major; no CVE driving it, deferred |
| `antlr4-python3-runtime` (Python, transitive via `omegaconf`←`rapidocr`←`docling`) | 4.9.3 | 4.13.2 | Large jump, transitive-pinned by `omegaconf`'s compatibility range — not touched directly, same category of risk as `opencv-python` |
| `ast-serialize` (Python, transitive via `mypy`, dev-only) | 0.6.0 | 0.8.0 | 0.x "zero-ver" jump, breaking-change-equivalent per this project's own convention (see `eslint-plugin-react-refresh` below) — dev-only, low risk, but held back pending a deliberate bump |
| `globals` (frontend) | 17.7.0 | 17.11.0 | Major — Dependabot PR **#80** covers 17.7.0→17.9.0 only; a residual gap to 17.11.0 will remain even after that PR merges — flagging for a follow-up bump next cycle |
| `vite` (frontend) | 8.2.0 | 8.2.1 | Patch — already covered by open Dependabot PR **#79**, not duplicated here |
| `typescript` (frontend) | 6.0.3 | 7.0.2 | Major (TS7 "Corsa" native compiler rewrite) — **no longer covered by any open Dependabot PR** (the prior PR #66 that tracked this is gone from the open list, presumably closed without merging) — re-flagging as an outstanding, currently-untracked major |
| `eslint-plugin-react-refresh` (frontend) | 0.4.26 | 0.5.4 | 0.x "minor" that is breaking-change-equivalent per semver-zero convention — **no longer covered by any open Dependabot PR** (prior PR #72 is gone from the open list) — re-flagging as outstanding |

### Post-change verification
All checks re-run after dependency bumps, everything green:
- `uv run pytest tests/ -v` → **177 passed**
- `uv run ruff check .` → clean
- `uv run black --check .` → clean
- `uv run isort --check .` → clean
- `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip` → clean
- `uv run pre-commit run --all-files` → clean (black, isort)
- `npm -C frontend run test` → **29 passed (5 files)**
- `npm -C frontend run lint` → clean
- `npm -C frontend run build` → clean
- `npm -C frontend audit` → 0 vulnerabilities (down from 1)
- Confirmed via `git diff --stat` that only `uv.lock` and `frontend/package-lock.json` changed — no `pyproject.toml` or `package.json` range edits were needed, and no env vars were added (`.env.example` untouched)

---

## 2026-08-05

> **⚠️ Open-PR backlog — verified fresh via `mcp__github__list_pull_requests` (state=open), not assumed from any prior note.** 12 PRs are currently open against this repo. Against `staging`: **#74** (prior weekly-maintenance cycle, "2026-07-29") is still open/unmerged; **#71** (docs: fix documentation drift) is still open; and 9 Dependabot PRs are open — **#73** (`jsdom` 28.1.0→30.0.0, now a bigger major than the 29.1.1 seen two cycles ago), **#72** (`eslint-plugin-react-refresh` 0.4.26→0.5.3), **#69** (`@testing-library/jest-dom` 6.9.1→7.0.0), **#67** (`globals` 16.5.0→17.7.0), **#66** (`typescript` 6.0.3→7.0.2), and GitHub Actions bumps **#65** (`actions/checkout` 6→7), **#64** (`actions/github-script` 7→9), **#63** (`actions/setup-node` 6→7), **#62** (`actions/setup-python` 6→7). Separately, **#75** (docs fix) is open against `master` directly. None of #74's dependency bumps have landed on `staging`, so — same as every prior cycle — this audit was run against the real, current `origin/staging` tip (`1a4a6ce`), not assumed-merged prior work. **Every package covered by an open Dependabot PR above was explicitly excluded from this cycle's update pass** to avoid duplicate/conflicting bumps landing out of order (see Dependency updates below for confirmation none of those five npm packages were touched here). Recommend the user merge or close #74, #71, #75, and the Dependabot backlog before the next cycle — this note has repeated for three cycles running now and the backlog keeps growing.

### Checks performed
- Fetched `origin/staging`, diffed local branch both directions (`git log claude/modest-faraday-f8vh9h..origin/staging` / reverse) — confirmed no non-merge commits unique to the local branch, then reset the working branch to `origin/staging`'s actual tip (`1a4a6ce`)
- Verified the open-PR backlog live via GitHub MCP tools (see banner above) rather than trusting any carried-over list
- Baseline **before any changes**: `uv sync --all-extras && uv run pytest tests/ -v` → **177 passed**
- Baseline: `uv run ruff check .`, `uv run black --check .`, `uv run isort --check .`, `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip`, `uv run pre-commit run --all-files` → all clean, no pre-existing issues
- Baseline: `npm -C frontend ci && npm -C frontend run test` → **29 passed (5 files)**; `npm -C frontend run lint` and `npm -C frontend run build` → clean
- Python dependency audit: `uv pip list --outdated` cross-checked with `uv tree --invert` for transitively-constrained packages (e.g. confirmed `cryptography` and `aiohttp` are transitive-only, via `google-auth` and `kubernetes`/`langchain-community` respectively)
- Frontend dependency audit: `npm -C frontend outdated`, excluding the 5 packages already covered by open Dependabot PRs (#73, #72, #69, #67, #66)
- Security audit: `uv export --format requirements-txt --no-hashes` + `uv run --with pip-audit pip-audit -r <export>` (project's own Python 3.12 interpreter, not a fresh resolve — same environment-scan approach as prior cycles, since resolving the raw export against the live PyPI index still fails on point releases newer than what's published upstream); `npm -C frontend audit --json`
- Re-verified the chromadb `PYSEC-2026-311` finding against the current `ingestion.py`: grepped for `Chroma`/`HttpClient`/`PersistentClient`/`trust_remote_code` — still only `chromadb.PersistentClient` and `langchain_chroma.Chroma` embedded usage, no server component, no `trust_remote_code`. Finding still present upstream (confirmed via `pip index versions chromadb`, latest is still 1.5.9) and still not reachable in this app.

### Fixes applied
No pre-existing failures found — baseline was fully green (177 pytest, all lint/format/type checks clean, frontend 29 tests/lint/build clean). Applied targeted dependency bumps and security fixes only (see below).

### Dependency updates

**Python backend** — applied via `uv lock --upgrade-package <name>` one package at a time (never a blanket `uv lock --upgrade`), all within `pyproject.toml`'s existing `>=` floors — **no `pyproject.toml` edits were needed**:

| Package | Before | After | Type | Notes |
|---|---|---|---|---|
| `aiohttp` | 3.14.1 | 3.14.3 | patch | fixes 3 CVEs, see Security findings |
| `fastapi` | 0.139.2 | 0.141.1 | minor | |
| `langchain-core` | 1.5.0 | 1.5.3 | patch | |
| `langchain-google-genai` | 4.3.1 | 4.3.2 | patch | |
| `langchain-openai` | 1.4.0 | 1.4.1 | patch | |
| `langgraph` | 1.2.9 | 1.2.10 | patch | |
| `langsmith` | 0.9.5 | 0.10.16 | minor (transitive) | |
| `uvicorn` | 0.51.0 | 0.52.1 | minor | |
| `redis` | 8.0.1 | 8.1.0 | minor | |
| `docling` | 2.114.0 | 2.118.0 | minor | confirmed via lockfile diff that `opencv-python` stayed at `4.13.0.92` — same trap avoided as every prior cycle |
| `docling-core` | 2.87.1 | 2.90.0 | minor | pulled transitively by docling bump |
| `docling-slim` | 2.114.0 | 2.118.0 | minor | pulled transitively by docling bump |
| `docling-parse` | 7.4.0 | 7.10.0 | minor | |
| `doclang` | 0.7.0 | 0.7.3 | patch | dropped an unused optional dep (`saxonche`) upstream — not our change |
| `openai` | 2.47.0 | 2.53.0 | minor | |
| `google-auth` | 2.55.1 | 2.56.2 | minor | |
| `google-genai` | 2.10.0 | 2.16.0 | minor | |
| `huggingface-hub` | 1.21.0 | 1.26.0 | minor | |
| `hf-xet` | 1.5.1 | 1.6.0 | minor | |
| `grpcio` | 1.81.1 | 1.83.0 | minor | |
| `kubernetes` | 36.0.2 | 36.0.3 | patch | |
| `onnxruntime` | 1.27.0 | 1.28.0 | minor | |
| `numpy` | 2.5.0 | 2.5.1 | patch | |
| `pandas` | 3.0.3 | 3.0.5 | patch | |
| `pypdfium2` | 5.11.0 | 5.12.1 | minor | |
| `pylatexenc` | 2.10 | 2.11 | minor | |
| `filelock` | 3.29.4 | 3.32.2 | minor | |
| `fsspec` | 2026.6.0 | 2026.7.0 | minor | |
| `soupsieve` | 2.8.4 | 2.9.1 | minor | |
| `starlette` | 1.3.1 | 1.4.1 | minor | |
| `regex` | 2026.6.28 | 2026.7.19 | minor | |
| `typing-extensions` | 4.15.0 | 4.16.0 | minor | |
| `typer` | 0.24.2 | 0.26.8 | minor | |
| `platformdirs` | 4.10.0 | 4.11.0 | minor | |
| `pydantic-core`, `tokenizers`, `mpmath`, `antlr4-python3-runtime` | — | — | — | already pulled to latest transitively by the bumps above; no separate action needed |
| `uuid-utils` | 0.16.2 | 0.17.0 | minor | |
| `python-discovery` | 1.4.2 | 1.5.1 | minor | |
| `mail-parser` | 4.4.0 | 4.5.0 | minor | |
| `faker` (dev) | 40.28.0 | 40.36.0 | minor | |
| `colorlog` | 6.10.1 | 6.12.0 | minor | |
| `annotated-types` | 0.7.0 | 0.8.0 | minor | |
| `annotated-doc` | 0.0.4 | 0.0.5 | patch | |
| `anyio` | 4.14.1 | 4.14.2 | patch | |
| `certifi` | 2026.6.17 | 2026.7.22 | patch | |
| `cffi` | 2.0.0 | 2.1.1 | minor | |
| `charset-normalizer` | 3.4.7 | 3.4.9 | patch | |
| `greenlet` | 3.5.3 | 3.5.4 | patch | |
| `xxhash` | 3.8.0 | 3.8.1 | patch | |
| `yarl` | 1.24.2 | 1.24.5 | patch | |
| `types-requests` (dev) | 2.33.0.20260518 | 2.33.0.20260712 | patch | |
| `virtualenv` (dev) | 21.5.1 | 21.7.1 | minor | |
| `ruff` (dev) | 0.15.22 | 0.16.1 | minor | |
| `tqdm` | 4.68.3 | 4.70.0 | minor | |
| `opentelemetry-api`/`-sdk`/`-proto`/`-exporter-otlp-proto-common`/`-exporter-otlp-proto-grpc`/`-semantic-conventions` | 1.43.0 (0.64b0) | 1.44.0 (0.65b0) | minor | bumped as a set |

**Frontend** — applied via `npm -C frontend update` (package.json semver ranges unchanged, `package-lock.json` refreshed only), explicitly excluding the 5 packages already covered by open Dependabot PRs (`jsdom`, `eslint-plugin-react-refresh`, `@testing-library/jest-dom`, `globals`, `typescript`):

| Package | Before | After | Type |
|---|---|---|---|
| `@playwright/test` / `playwright` / `playwright-core` | 1.61.1 | 1.62.1 | patch |
| `@testing-library/user-event` | 14.6.1 | 14.6.3 | patch |
| `@types/react` | 19.2.17 | 19.2.18 | patch |
| `@types/react-dom` | 19.2.3 | 19.2.4 | patch |
| `@typescript-eslint/eslint-plugin` | 8.65.0 | 8.66.0 | minor |
| `@typescript-eslint/parser` | 8.65.0 | 8.66.0 | minor |
| `@vitejs/plugin-react` | 6.0.4 | 6.0.5 | patch |
| `eslint` | 10.7.0 | 10.8.0 | minor |
| `vite` | 8.1.5 | 8.2.0 | minor |
| `brace-expansion` (transitive, via `eslint`→`minimatch`) | 5.0.7 | 5.0.9 | patch — via `npm audit fix`, security fix |
| `postcss` (transitive, via `vite`) | 8.5.22 | 8.5.25 | patch — via `npm audit fix`, security fix |
| `undici` (transitive, via `jsdom`) | 7.28.0 | 7.29.0 | patch — via `npm audit fix`, security fix (fixed without touching `jsdom` itself, which stays on the Dependabot PR #73 track) |

### Security findings

**`npm -C frontend audit`** — before fixes: **3 vulnerabilities (1 moderate, 2 high)**, all with non-breaking fixes available via `npm audit fix` (no `package.json` range edits, no major bumps pulled in):

| Package | Version | Severity | Advisory | Resolution |
|---|---|---|---|---|
| `brace-expansion` | 5.0.7 | High | GHSA-mh99-v99m-4gvg / GHSA-rgw5-rvv9-x895 — DoS via unbounded expansion/intermediate arrays | **Fixed** — 5.0.9 |
| `postcss` | 8.5.22 | Moderate | GHSA-fxqj-rqcc-2cmp — incomplete fix of prior sourceMappingURL path-read issue | **Fixed** — 8.5.25 |
| `undici` | 7.28.0 | High | GHSA-4cwx-7wf7-3272 (+3 related moderate advisories) — cross-user cache info disclosure / request smuggling in Node's fetch stack | **Fixed** — 7.29.0 (patch bump only; did not require touching `jsdom`, which is intentionally left to Dependabot PR #73) |

After fixes: **`npm -C frontend audit` → 0 vulnerabilities.**

**Python (`pip-audit` against the exported `uv.lock` reqs, resolved with the project's own Python 3.12 interpreter via `uv run --with pip-audit`)** — before fixes: **5 known vulnerabilities in 3 packages**:

| Package | Version | ID | Severity/notes | Resolution |
|---|---|---|---|---|
| `aiohttp` | 3.14.1 | PYSEC-2026-3545 | Out-of-bounds heap read in the C response parser building an error message for a malformed response — client-side DoS | **Fixed** — upgraded to 3.14.3 |
| `aiohttp` | 3.14.1 | PYSEC-2026-3546 | Request-smuggling risk via a WebSocket-upgrade edge case in the HTTP parsers | **Fixed** — upgraded to 3.14.2+ |
| `aiohttp` | 3.14.1 | PYSEC-2026-3547 | Client decompresses RSV1-flagged frames even without `permessage-deflate` negotiated — CPU/memory amplification risk | **Fixed** — upgraded to 3.14.2+ |
| `chromadb` | 1.5.9 | PYSEC-2026-311 | Pre-auth code injection via `trust_remote_code` on the Chroma HTTP server's `/api/v2/.../collections` endpoint | **Not fixable** — 1.5.9 is still the latest release on PyPI as of this cycle (re-confirmed via `pip index versions chromadb`); no patched version exists |
| `cryptography` | 49.0.0 | PYSEC-2026-3552 | Bleichenbacher-style timing/error oracle in `pkcs7_decrypt_der`/`_pem`/`_smime` against attacker-supplied `EnvelopedData` — requires an app that auto-decrypts untrusted S/MIME and reflects the outcome | **Fix requires a major bump (50.0.0)** — see Major upgrades table; not applied this cycle, see risk assessment below |

**Risk assessment — chromadb PYSEC-2026-311 (remaining, unpatched upstream, re-verified this cycle):** re-grepped `ingestion.py` and `graph/` for `Chroma`/`HttpClient`/`PersistentClient`/`trust_remote_code` — the code still only ever constructs `chromadb.PersistentClient` and `langchain_chroma.Chroma` as an embedded, on-disk, in-process client against `_CHROMA_DIR`. No standalone Chroma HTTP/gRPC server is started anywhere in this codebase, and no code path sets `trust_remote_code`. **Still assessed as not reachable in this app's current usage.** Same conclusion as the 2026-07-08/07-15/07-22 cycles; will keep re-checking each cycle (and via `weekly-audit.yml`) until a fix ships.

**Risk assessment — cryptography PYSEC-2026-3552 (fix exists at 50.0.0, held back as a major):** the vulnerable functions (`pkcs7_decrypt_der`/`_pem`/`_smime`) decrypt attacker-supplied S/MIME `EnvelopedData` and are only exploitable by an application that does that and reflects the outcome (e.g. a mail gateway). Grepped the codebase for `cryptography`/`pkcs7` usage directly — none found; `cryptography` is a transitive-only dependency pulled in by `google-auth` (for JWT/service-account signing), which never touches PKCS#7/S-MIME. **Assessed as not reachable in this app.** Flagged for a dedicated `cryptography` major-version review rather than silently applied, per this cycle's policy of holding back majors even when they carry a security fix, given the low practical exploitability here.

### Major upgrades — flagged, NOT applied

| Package | Current | Available | Why held back |
|---|---|---|---|
| `cryptography` (Python, transitive via `google-auth`) | 49.0.0 | 50.0.0 | Major version; fixes PYSEC-2026-3552 but assessed not reachable in this app (see risk assessment above) — needs a dedicated review rather than a drive-by major bump |
| `opencv-python` (Python, transitive via `docling`/`rapidocr`) | 4.13.0.92 | 5.0.0.93 | Major; known breaking-change risk for docling's OCR path — held back every cycle since 2026-07-08, re-confirmed this cycle that the `docling` minor bump does not pull it in |
| `semchunk` (Python, transitive via `docling`) | 3.2.5 | 4.1.1 | Major; carried over from the 2026-07-15 cycle |
| `websockets` (Python, transitive) | 15.0.1 | 17.0.1 | Major (two majors behind); carried over, no security relevance found this cycle |
| `transformers` (Python, transitive via `docling`/`torch`) | 5.8.1 | 5.14.1 | Large jump (12+ releases) tied to `docling`/`torch` compatibility — deferred pending dedicated review, carried over from 2026-07-15 |
| `@testing-library/jest-dom` (frontend) | 6.9.1 | 7.0.0 | Major — **already covered by open Dependabot PR #69**, not duplicated here |
| `globals` (frontend) | 16.5.0 | 17.7.0 | Major — **already covered by open Dependabot PR #67**, not duplicated here (flagged every cycle since 2026-06-17) |
| `jsdom` (frontend) | 28.1.0 | 30.0.1 | Major — **already covered by open Dependabot PR #73** (now targeting 30.0.0, a bigger jump than the 29.1.1 seen at the 2026-07-15 cycle); not duplicated here. The `undici` security fix above was applied independently without needing this bump. |
| `eslint-plugin-react-refresh` (frontend) | 0.4.26 | 0.5.3 | 0.x "minor" that is breaking-change-equivalent per semver-zero convention — **already covered by open Dependabot PR #72**, not duplicated here |
| `typescript` (frontend) | 6.0.3 | 7.0.2 | Major (TS7 "Corsa" native compiler rewrite) — **already covered by open Dependabot PR #66**, not duplicated here |
| `torch` / `torchvision` | 2.13.0 / 0.28.0 | — (current) | Not flagged this cycle — already at latest, no action needed |

### Post-change verification
All checks re-run after dependency bumps, everything green:
- `uv run pytest tests/ -v` → **177 passed**
- `uv run ruff check .` → clean
- `uv run black --check .` → clean
- `uv run isort --check .` → clean
- `uv run mypy api/main.py api/rate_limit.py tests/test_api.py --follow-imports=skip` → clean
- `uv run pre-commit run --all-files` → clean (black, isort)
- `npm -C frontend run test` → **29 passed (5 files)**
- `npm -C frontend run lint` → clean
- `npm -C frontend run build` → clean
- `npm -C frontend audit` → 0 vulnerabilities (down from 3)
- Confirmed via diff that only `uv.lock` and `frontend/package-lock.json` changed — no `pyproject.toml` or `package.json` range edits were needed, and no env vars were added (`.env.example` untouched)

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
## 2026-07-15

### Checks performed
- Branch created from latest `origin/staging` tip (`885e373`). Note: PR #57 (2026-07-08 maintenance) and PR #58 (docs) are still open/unmerged against `staging`, so this cycle's "before" state does not include their changes — a fresh dependency audit was run rather than assuming #57's bumps already landed.
- Baseline (before any changes):
  - `uv sync --all-extras` + `uv run pytest tests/ -v` → **177 passed**
  - `npm -C frontend ci` + `npm -C frontend run test` → **29 passed**; `npm run lint` and `npm run build` clean
  - No pre-existing test/lint/build failures found.
- CI health: last workflow runs directly on `staging` (`chore: merge all open PRs (#35–#55) into master`, 2026-07-06T11:49:57Z and the preceding weekly-maintenance-check run) are all `completed` / `success`. Staging CI is green.
- `uv pip list --outdated`, `npm -C frontend outdated` reviewed for the full dependency surface.

### Fixes applied

**Python** (`uv.lock`, applied with `uv lock --upgrade-package <name>` per package, one command covering all safe targets at once to avoid a blanket `uv lock --upgrade` — same opencv-python trap called out in the 2026-07-08 cycle was checked for and did **not** trigger this time):

| Package | Before | After |
|---|---|---|
| `docling` | 2.108.0 | 2.113.0 |
| `docling-core` | 2.85.0 | 2.87.1 |
| `docling-slim` | 2.108.0 | 2.113.0 |
| `rapidocr` | 3.9.0 | 3.9.1 |
| `langchain` | 1.3.11 | 1.3.13 |
| `langchain-core` | 1.4.8 | 1.4.9 |
| `langchain-google-genai` | 4.2.6 | 4.2.7 |
| `langchain-mistralai` | 1.1.5 | 1.1.6 |
| `langchain-openai` | 1.3.3 | 1.3.5 |
| `langgraph` | 1.2.7 | 1.2.9 |
| `uvicorn` | 0.49.0 | 0.51.0 |
| `mypy` (dev) | 2.1.0 | 2.3.0 |
| `ruff` (dev) | 0.15.20 | 0.15.21 |
| `anyio` | 4.14.1 | 4.14.2 |
| `cffi` | 2.0.0 | 2.1.0 |
| `charset-normalizer` | 3.4.7 | 3.4.9 |
| `filelock` | 3.29.4 | 3.29.7 |
| `google-auth` | 2.55.1 | 2.56.0 |
| `google-genai` | 2.10.0 | 2.11.0 |
| `grpcio` | 1.81.1 | 1.82.1 |
| `huggingface-hub` | 1.21.0 | 1.23.0 |
| `kubernetes` | 36.0.2 | 36.0.3 |
| `numpy` | 2.5.0 | 2.5.1 |
| `openai` | 2.44.0 | 2.45.0 |
| `pyasn1` | 0.6.3 | 0.6.4 |
| `python-discovery` | 1.4.2 | 1.4.4 |
| `regex` | 2026.6.28 | 2026.7.10 |
| `tqdm` | 4.68.3 | 4.68.4 |
| `types-requests` (dev) | 2.33.0.20260518 | 2.33.0.20260712 |
| `typing-extensions` | 4.15.0 | 4.16.0 |
| `xxhash` | 3.8.0 | 3.8.1 |
| `librt` (transitive) | 0.12.0 | 0.13.0 |

`pyproject.toml` floors were left unchanged — every new locked version already satisfies the existing `>=` constraints, so no source edits were needed.

**Frontend** (`frontend/package-lock.json` only, via `npm -C frontend update` — `package.json` semver ranges unchanged, pure patch/minor lockfile refresh):

| Package | Before | After |
|---|---|---|
| `@typescript-eslint/eslint-plugin` | 8.62.1 | 8.64.0 |
| `@typescript-eslint/parser` | 8.62.1 | 8.64.0 |
| `eslint` | 10.6.0 | 10.7.0 |
| `vite` | 8.1.2 | 8.1.4 |
| `vitest` | 4.1.9 | 4.1.10 |
| assorted transitive patch bumps | — | — |

### Not applied / deferred

- **`setuptools` 81.0.0 → 83.0.0`: fix exists for a flagged pip-audit finding (`PYSEC-2026-3447`) but `uv lock --upgrade-package setuptools` proved it is transitively pinned by `torch==2.12.1` (`setuptools<82`). Forcing it would require also bumping `torch` (heavy CUDA/ML dependency, itself only a minor bump to 2.13.0 but with a large blast radius for an OCR/embedding pipeline) — treated as out of scope for a build-tool-only CVE this cycle. Risk assessment: `setuptools` here is only a build-backend requirement (`[build-system] requires = ["setuptools>=61.0"]`) resolved from a trusted, pinned lockfile; it is not exposed to untrusted/attacker-controlled package metadata at runtime, so the practical exploitability in this deployment is low. Flagged for the next cycle to revisit alongside a deliberate `torch` bump.

### Security findings

- **`npm -C frontend audit`: 0 vulnerabilities.**
- **Python (`pip-audit` against exported `uv.lock` reqs, resolved with the project's own Python 3.12 interpreter via `uv run --with pip-audit`): 2 findings.**
  - **CRITICAL — still un-applied, no fix exists yet:** `chromadb` **1.5.9** (current latest release on PyPI) — **PYSEC-2026-311 / CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c ("ChromaToast")**, a pre-authentication RCE in the `/api/v2/tenants/{tenant}/databases/{db}/collections` endpoint via a malicious embedding-function model reference with `trust_remote_code=true`. This is the same finding documented in the 2026-07-08 log (PR #57) — re-verified this cycle via `pip index versions chromadb`: still no patched release on PyPI.
    - **Risk assessment (unchanged):** this app only ever constructs `langchain_chroma.Chroma(...)` as an embedded/persistent client (`ingestion.py`, `graph/`) — it never starts chromadb's standalone HTTP server (`chroma run` / `HttpClient`) and never sets `trust_remote_code`. The vulnerable server endpoint is not exposed by this deployment. Vulnerable code still ships in the dependency; tracked for a future cycle, weekly-audit.yml will keep re-checking automatically.
  - **`setuptools` 81.0.0 — `PYSEC-2026-3447`, fix at 83.0.0** — see "Not applied / deferred" above for why this was not force-applied this cycle.

### Compatibility check (post-change verification, all green)
- `uv run pytest tests/ -v` → 177 passed (no change from baseline)
- `uv run ruff check .`, `uv run black --check .`, `uv run isort --check .` → all clean
- `uv run pre-commit run --all-files` (black, isort hooks) → Passed
- `npm -C frontend run test` → 29 passed, `npm -C frontend run lint` and `npm -C frontend run build` → clean

### Known pre-existing issue (not introduced this cycle)
- `uv run mypy api/main.py api/rate_limit.py tests/test_api.py` reports **40 errors in 18 files**. Verified by temporarily reverting `uv.lock` to the untouched `staging` baseline and re-running with the original `mypy==2.1.0` — the identical 40 errors are already present before any dependency change this cycle made. Mostly `langchain`/`langgraph` `Runnable`/`BaseChatModel` typing mismatches (`graph/chains/*`, `graph/nodes/*`, `graph/graph.py`) plus a couple of unrelated `attr-defined`/`assignment` issues (`ingestion.py`, `document_updates.py`, `build_document_sources.py`, `api/rate_limit.py`). Not caused by this cycle's bumps and not fixed here (would require a dedicated typing pass, out of scope for a weekly dependency/security cycle) — flagged for a future cycle.

### Major upgrades pending (not applied — flagged for manual review)

| Package | Current | Available | Why held back |
|---|---|---|---|
| `chromadb` | 1.5.9 | — (unpatched) | See security findings above |
| `setuptools` | 81.0.0 | 83.0.0 | Security fix exists but pinned `<82` transitively by `torch==2.12.1`; would require a coordinated `torch` bump |
| `typescript` (frontend, dev) | 6.0.3 | 7.0.2 | Major — TS7 ("Corsa") is a from-scratch native/Go compiler port, needs dedicated review |
| `jsdom` (frontend, dev) | 28.1.0 | 29.1.1 | Major |
| `globals` (frontend, dev) | 16.5.0 | 17.7.0 | Major, carried over from the 2026-06-17 cycle |
| `eslint-plugin-react-refresh` (frontend, dev) | 0.4.26 | 0.5.3 | Pre-1.0 (`0.x`) bump outside current caret range — treated as breaking-risk per semver convention for `0.x` packages |
| `semchunk` | 3.2.5 | 4.1.1 | Major, transitive via `docling` |
| `opencv-python` | 4.13.0.92 | 5.0.0.93 | Major, transitive via `docling`/`rapidocr` — explicitly avoided (the known blanket-upgrade trap from the 2026-07-08 cycle) |
| `websockets` | 15.0.1 | 16.1 | Major, transitive |
| `torch` / `torchvision` | 2.12.1 / 0.27.1 | 2.13.0 / 0.28.0 | Minor, but heavy CUDA/ML dependency pair with a large blast radius (also the blocker for the `setuptools` security fix above) — deferred to a dedicated cycle |
| `transformers` | 5.8.1 | 5.14.0 | Large minor jump (6 releases), tied to `docling`/`torch` compatibility — deferred pending dedicated review |

### Not touched
PR #57 (2026-07-08 maintenance) and PR #58 (docs) — both open against `staging`, left completely untouched per instructions. PR #56 (`staging` → `master`) also untouched.
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
