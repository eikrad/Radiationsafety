# Changelog

All notable changes to this project are documented in this file.

<!-- Versions from 0.5.0 on are written by release-please from Conventional Commit
     messages (see docs/releasing.md). Notes that commits cannot carry,
     such as upgrade steps, are added by hand below the generated entry. -->

## 0.5.0 upgrade notes

Read these before upgrading from 0.4.x. The generated list of features and fixes
for 0.5.0 is above.

### Changed
- **⚠️ Breaking: Scaleway is the default provider** for answers (`gemma-4-26b-a4b-it`) and
  retrieval embeddings (`bge-multilingual-gemma2`), EU-hosted, instead of Gemini. On the golden
  set, BGE embeddings found more of the relevant passages than Gemini (evidence recall 0.90 vs
  0.81, pass rate 79 % vs 71 %). A setup without `LLM_PROVIDER` / `EMBEDDING_PROVIDER` now needs
  `SCW_SECRET_KEY` and the Scaleway collections
  (`EMBEDDING_PROVIDER=scaleway uv run python ingestion.py --reembed-from gemini`); set both to
  `gemini` to keep the previous behaviour. Gemini, OpenAI, Mistral and Ollama stay available.
- The UI offers Scaleway first, with its models from the server's `.env`, and a Scaleway key
  field in Settings. The privacy notice no longer says every question goes to Google.

### Privacy and compliance (#94)
- **Fixed a Privacy Mode leak:** with `LLM_PROVIDER=ollama`, the generation-retry path
  could still fall back to Brave web search after two failed retries. It now ends
  instead, and `web_search` refuses to run in Privacy Mode as a second guard.
- LangSmith tracing is off by default; when enabled, the EU endpoint is the documented
  default.
- The header shows a persistent AI disclosure and a not-legal/clinical-advice notice
  (EU AI Act Art. 50(1)); a new Privacy notice names the controller from the optional
  `PRIVACY_CONTROLLER_NAME` / `PRIVACY_CONTROLLER_CONTACT` variables.
- `X-Forwarded-For` is trusted only with `TRUST_PROXY_HEADERS=true`; before, any client
  could spoof it to get around rate limits.
- API keys entered in the browser live in `sessionStorage` instead of `localStorage`.
- Three Danish bekendtgørelser in force since 1 January 2026 (BEK 1386–1388) are staged;
  run `uv run python ingestion.py` to embed them.

### Setup checks (#115)
- Settings show per provider whether the server can answer with it and, if not, what is
  missing (e.g. `SCW_MODEL`, `SCW_EMBED_MODEL`, an unbuilt search index). A misconfigured
  provider now returns a 503 with that reason instead of a bare 500.
- **Check your `.env` after upgrading:** with `EMBEDDING_PROVIDER` unset, every cloud
  provider searches with Scaleway embeddings. Set `EMBEDDING_PROVIDER=gemini` to keep
  using an existing Gemini index.

### Security
- **`cryptography` 49.0.0 → 50.0.1 (⚠️ major version bump)** — fixes
  [GHSA-g6cj-pr64-35w5](https://github.com/advisories/GHSA-g6cj-pr64-35w5)
  (PYSEC-2026-3552, CVE-2026-69247): `pkcs7_decrypt_der`/`pkcs7_decrypt_pem`/
  `pkcs7_decrypt_smime` leaked a Bleichenbacher timing/output oracle against
  the recovered content-encryption key (introduced in 44.0.0, fixed in 50.0.0).
  `cryptography` is a transitive dependency (via `google-auth` →
  `google-genai` → `langchain-google-genai`), not pinned directly in
  `pyproject.toml`; bumped via `uv lock --upgrade-package cryptography`.
  Verified with `pip-audit` (CVE no longer reported) and the full test/lint
  suite (177 backend tests, ruff, black, isort — all green; no code changes
  required, `uv.lock` only).

### Notes (routine weekly maintenance, no code changes otherwise)
- `pip-audit` against the resolved environment also flagged `transformers`
  5.8.1 (CVE-2026-9856, path-traversal in `save_pretrained`, fixed in
  5.10.0) and `accelerate`/`chromadb` (no fix version published yet).
  `transformers` could not be bumped: `docling-core`/`docling-ibm-models`
  cap it at `<5.9.0` on `sys_platform == "darwin"`, and `uv.lock` is a
  cross-platform lock, so `uv lock --upgrade-package transformers` cannot
  select 5.10.0 without breaking macOS installs. Left at 5.8.1 pending an
  upstream `docling` release that relaxes the darwin cap; tracked for a
  future maintenance pass rather than forced here.

## 0.4.0 - 2026-06-11

### Added
- **Privacy Mode**: fully air-gapped operation via Ollama for both LLM generation and embeddings.
  - No cloud API calls, LangSmith tracing, or web search when `LLM_PROVIDER=ollama`
  - Separate Chroma collections (`-ollama` suffix) to preserve cloud indexes
  - Automatic privacy guards: web search and tracing disabled in privacy mode
  - Local embedding models: `nomic-embed-text` (default)
  - Local LLM models: `llama3.1:8b` (default)
  - Configurable via `.env`: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`
- **Test parallelization**: pytest-xdist for parallel test execution (`-n auto` in CI)
- Updated all dependency versions to currently installed stable versions for consistency

### Changed
- CI now runs tests in parallel on available CPU cores for faster feedback
- All main and dev dependencies pinned to stable versions (e.g., chromadb 1.5+, fastapi 0.136+, langchain 1.3+, pytest 9.0+)
- `grade_documents` node now respects `privacy_mode` flag to prevent web search in air-gapped mode

## 0.3.0 - 2026-05-18

### Added
- **Reflexion retry loop** (Shinn et al., NeurIPS 2023): when a generation fails
  grading, the grader now produces a short verbal hint (`missing_info`) describing
  the specific missing fact or document section. This hint is stored as `reflection`
  in graph state and passed to `retrieve_missing` on the next attempt, so the
  retrieval query targets the exact gap rather than blindly re-querying.
- New `GRADE_GENERATION` node (`graph/nodes/grade_generation.py`): extracts the
  LLM grading work from the old combined routing function into a proper LangGraph
  node so it can write `reflection` and `generation_passed_grading` to state.
- New state fields: `reflection: str`, `generation_passed_grading: bool`.
- `reflection` parameter on `invoke_missing_query_chain` — when non-empty, the
  hint is injected into the human prompt turn as focused context.
- Two new test files: `tests/test_grade_generation.py`, `tests/test_reflection.py`.

### Changed
- `GradeGeneration` schema simplified: `grounded: bool` + `answers_question: bool`
  collapsed into `passed: bool` + `missing_info: str`. Both old fields routed
  identically; merging them makes the grader prompt cleaner and less ambiguous.
- `grade_generation_grounded` routing function split into:
  - `GRADE_GENERATION` node (LLM call, writes state)
  - `route_after_grade_generation` (pure function, no LLM call, reads state flags)
- `generate` node now resets `reflection = ""` on every generation attempt so
  stale hints never leak into the next turn of a multi-turn conversation.
- `eval/metrics.py`: `faithfulness` and `answer_relevance` now both read `passed`
  (previously read `grounded` and `answers_question` respectively).
- Test count: 126 → 135.

## 0.2.0 - 2026-04-30

### Added
- Admin-token protection for mutating backend routes with fail-closed behavior.
- In-memory per-client rate limiting for query/admin endpoints with `Retry-After` on `429`.
- Optional Redis rate-limit backend for multi-replica deployments (`RATE_LIMIT_BACKEND=redis`).
- Request correlation via `X-Request-ID`.
- Expanded Prometheus-style metrics:
  - request totals and error totals,
  - duration sum,
  - per-endpoint request/error counters,
  - response status-class counters,
  - web-search attempt counter.
- Pre-commit hook configuration (`black --check`, `isort --check-only`) and CI alignment.

### Changed
- CI now validates pre-commit checks and frontend build before tests complete.
- Docker/Compose runtime hardening:
  - non-root backend runtime,
  - healthchecks and startup dependency on healthy backend,
  - reduced privileges/capabilities in Compose,
  - safer runtime defaults and graceful shutdown tuning.

### Notes
- Mypy gate is intentionally scoped in CI to changed critical files and can be expanded gradually.
