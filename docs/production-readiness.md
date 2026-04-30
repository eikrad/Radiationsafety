# Production Readiness Baseline

## Route Exposure Matrix

- Public routes: `/query`, `/health`, `/metrics`, `/config`, read-only document routes.
- Admin routes (token required): mutating document routes and `/ingest`.

## Admin Auth Default

- Fail-closed default: if `ADMIN_TOKEN` is not configured, admin routes return `503`.
- Valid admin requests must send `X-Admin-Token` header matching `ADMIN_TOKEN`.
- Optional local bypass: `ADMIN_AUTH_BYPASS=true` (intended for explicit local-only use).

## Rate Limiting MVP

- Initial MVP target: in-memory rate limiting for single-process deployments.
- Config knobs: `RATE_LIMIT_QUERY_MAX_REQUESTS`, `RATE_LIMIT_QUERY_WINDOW_SEC`, `RATE_LIMIT_ADMIN_MAX_REQUESTS`, `RATE_LIMIT_ADMIN_WINDOW_SEC`.
- Known limitation: not globally consistent across multiple workers/replicas.
- Future upgrade path: shared external store (for example Redis) if horizontal scaling is needed.

## Observability Baseline

- Every HTTP response includes `X-Request-ID` for correlation.
- `/metrics` exports request totals, error totals, request duration sum, and query web-search attempts.

## Test Priorities (TDD Order)

1. Admin auth behavior on mutating routes (`401`/`503`/success path).
2. Rate-limit behavior (`429` and success-under-limit).
3. Failure modes and stable error contracts.
