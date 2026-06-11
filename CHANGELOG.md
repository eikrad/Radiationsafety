# Changelog

All notable changes to this project are documented in this file.

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
