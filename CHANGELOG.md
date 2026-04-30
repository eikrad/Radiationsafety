# Changelog

All notable changes to this project are documented in this file.

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
