# Production Readiness

Operational reference for deploying and running the Radiation Safety RAG backend. Covers the security boundary between public and admin routes, authentication, rate limiting, and observability.

---

## Route Exposure Matrix

| Route | Visibility | Auth required |
|---|---|---|
| `GET /query` | Public | No |
| `GET /health` | Public | No |
| `GET /metrics` | Public | No |
| `GET /config` | Public | No |
| `GET /documents/*` (read-only) | Public | No |
| `POST /ingest` | Admin | `X-Admin-Token` |
| `POST /documents/*` (mutating) | Admin | `X-Admin-Token` |
| `DELETE /documents/*` | Admin | `X-Admin-Token` |

---

## Admin Authentication

- **Fail-closed default:** if `ADMIN_TOKEN` is not set, all admin routes return `503 Service Unavailable`.
- Valid requests must include the `X-Admin-Token` header with the value matching `ADMIN_TOKEN`.
- **Local bypass:** set `ADMIN_AUTH_BYPASS=true` to skip auth checks. Intended for local development only — never set this in production.

---

## Rate Limiting

Rate limits are enforced per-client (by IP or token) and apply to both public and admin endpoints.

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_QUERY_MAX_REQUESTS` | — | Max requests per window for `/query` |
| `RATE_LIMIT_QUERY_WINDOW_SEC` | — | Window duration in seconds for query rate limit |
| `RATE_LIMIT_ADMIN_MAX_REQUESTS` | — | Max requests per window for admin routes |
| `RATE_LIMIT_ADMIN_WINDOW_SEC` | — | Window duration in seconds for admin rate limit |
| `RATE_LIMIT_BACKEND` | `memory` | `memory` (single-process) or `redis` (multi-replica) |
| `RATE_LIMIT_REDIS_URL` | — | Redis connection URL when `RATE_LIMIT_BACKEND=redis` |

The in-memory backend is suitable for single-process deployments. For multi-worker or multi-replica setups, use `RATE_LIMIT_BACKEND=redis` to enforce global limits across all instances. A `429 Too Many Requests` response includes a `Retry-After` header.

---

## Observability

- Every HTTP response includes an `X-Request-ID` header for request correlation across logs.
- `GET /metrics` exports Prometheus-style counters:
  - Total request and error counts (overall and per endpoint)
  - Request duration sum (seconds)
  - Response status-class counters (2xx, 4xx, 5xx)
  - Web search attempt counter

---

## Test Coverage Priorities

Ordered by risk — these should be covered before adding new functionality:

1. Admin auth on mutating routes: `401` on bad token, `503` when `ADMIN_TOKEN` is absent, success path with valid token.
2. Rate-limit behavior: `429` when limit exceeded, success responses under the limit.
3. Failure modes: stable error contracts on invalid input and backend errors.
