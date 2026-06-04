# Production Readiness Baseline

## Route Exposure Matrix

- Public routes: `/query`, `/health`, `/metrics`, `/config`, read-only document routes.
- Admin routes (token required): mutating document routes and `/ingest`.

## Admin Auth Default

- Fail-closed default: if `ADMIN_TOKEN` is not configured, admin routes return `503`.
- Valid admin requests must send `X-Admin-Token` header matching `ADMIN_TOKEN`.
- Optional local bypass: `ADMIN_AUTH_BYPASS=true` (intended for explicit local-only use).

## Rate Limiting

- Default backend: in-memory rate limiting for single-process deployments.
- Optional multi-replica backend: set `RATE_LIMIT_BACKEND=redis` and `RATE_LIMIT_REDIS_URL`.
- Config knobs: `RATE_LIMIT_QUERY_MAX_REQUESTS`, `RATE_LIMIT_QUERY_WINDOW_SEC`, `RATE_LIMIT_ADMIN_MAX_REQUESTS`, `RATE_LIMIT_ADMIN_WINDOW_SEC`.

## Observability Baseline

- Every HTTP response includes `X-Request-ID` for correlation.
- `/metrics` exports Prometheus-style counters and sums:
  - `req_total` — total HTTP requests
  - `req_errors` — total HTTP errors
  - `req_duration_sum` — cumulative request duration (seconds)
  - `req_by_path[path]` — per-endpoint request counts
  - `req_errors_by_path[path]` — per-endpoint error counts
  - `req_status_class[2xx|4xx|5xx]` — response status-class breakdown
  - `web_search_attempts` — number of Brave Search fallback invocations

## Test Priorities (TDD Order)

1. Admin auth behavior on mutating routes (`401`/`503`/success path).
2. Rate-limit behavior (`429` and success-under-limit).
3. Failure modes and stable error contracts.
