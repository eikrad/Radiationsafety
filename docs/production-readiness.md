# Production Readiness

Reference for deploying and operating the Radiation Safety RAG system.

---

## Route exposure

| Route | Auth | Notes |
|---|---|---|
| `POST /query` | Public | Rate-limited |
| `GET /health` | Public | Container healthcheck target |
| `GET /metrics` | Public | Prometheus-style counters |
| `GET /config` | Public | Returns which LLM keys are configured |
| `GET /documents/check-updates` | Public | Polls retsinformation.dk / IAEA for newer versions |
| `POST /ingest` | **Admin** | Triggers full re-ingestion in background |
| `POST /documents/add-pdf` | **Admin** | Upload and register a new PDF |
| `PATCH /documents/source/{id}/url` | **Admin** | Manually update a source URL |
| `POST /documents/source/{id}/lookup-url` | **Admin** | Auto-resolve newest URL for a source |
| `POST /documents/source/{id}/download-update` | **Admin** | Download and apply the newest version |
| `POST /documents/build-from-local` | **Admin** | Rebuild registry from local PDFs |
| `POST /documents/sync-danish` | **Admin** | Sync all Danish sources to newest versions |

---

## Admin authentication

Admin routes require the `X-Admin-Token` header with a value matching the `ADMIN_TOKEN` environment variable.

**Fail-closed by default:** if `ADMIN_TOKEN` is not set, all admin routes return `503 Service Unavailable`.

```bash
# Good — admin token configured
ADMIN_TOKEN=your-secret-token

# Dev-only bypass (never use in production)
ADMIN_AUTH_BYPASS=true
```

---

## Rate limiting

| Env variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_BACKEND` | `in_memory` | `in_memory` or `redis` |
| `RATE_LIMIT_REDIS_URL` | — | Required if using Redis backend |
| `RATE_LIMIT_QUERY_MAX_REQUESTS` | `60` | Max query requests per window |
| `RATE_LIMIT_QUERY_WINDOW_SEC` | `60` | Window size in seconds |
| `RATE_LIMIT_ADMIN_MAX_REQUESTS` | `20` | Max admin requests per window |
| `RATE_LIMIT_ADMIN_WINDOW_SEC` | `60` | Admin window size in seconds |

**Single-process deployments:** `in_memory` is suitable. Limits are per-client (IP-based).

**Multi-worker / multi-replica deployments:** set `RATE_LIMIT_BACKEND=redis` and provide `RATE_LIMIT_REDIS_URL`. Without this, each worker has its own counter and limits are not enforced globally.

---

## Observability

- Every HTTP response includes an `X-Request-ID` header for log correlation.
- `GET /metrics` exports Prometheus-style counters, all prefixed `radiationsafety_`:
  - `radiationsafety_graph_loaded` — 1 if the RAG graph is loaded, 0 otherwise
  - `radiationsafety_uptime_seconds` — process uptime
  - `radiationsafety_http_requests_total` — total HTTP requests served
  - `radiationsafety_http_errors_total` — total HTTP responses with status >= 400
  - `radiationsafety_http_request_duration_seconds_sum` — cumulative request durations
  - `radiationsafety_query_web_search_attempts_total` — query runs that attempted web search
  - `radiationsafety_query_outcomes_total{outcome=...}` — query outcomes by routing category (see `routing_outcome` in [architecture.md](architecture.md))
  - `radiationsafety_http_requests_by_path_total{path=...}` — requests broken down by path
  - `radiationsafety_http_errors_by_path_total{path=...}` — errors broken down by path
  - `radiationsafety_http_responses_by_status_class_total{status_class=...}` — responses broken down by status class (e.g. `2xx`, `4xx`)

---

## Container hardening

The Docker setup (`Dockerfile` + `docker-compose.yml`) applies these defaults:

- Backend runs as non-root user `appuser`.
- `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1` are set.
- Compose applies `no-new-privileges: true` and `cap_drop: [ALL]` to backend and frontend containers.
- `/tmp` is a `tmpfs` mount (not persisted).
- Chroma data lives in a named volume (`chroma_data`) mounted at `/app/.chroma`.
- Backend healthcheck is active; frontend service waits for backend healthy before starting.

---

## Runbook

### 429 Too Many Requests spike

1. Check `RATE_LIMIT_QUERY_MAX_REQUESTS` and `RATE_LIMIT_QUERY_WINDOW_SEC`.
2. Identify the source IP from logs (each request logs its `X-Request-ID` and client IP).
3. If the rate is legitimate traffic, increase the limit or switch to Redis backend for global enforcement.

### Admin routes return 503

1. Verify `ADMIN_TOKEN` is set in the environment.
2. Confirm `ADMIN_AUTH_BYPASS` is not accidentally set to `false` while `ADMIN_TOKEN` is missing.
3. Restart the backend after fixing the environment.

### Health degraded

1. Check `GET /health` — returns `{"status": "ok"}` when healthy.
2. Check `GET /metrics` for error rate increases.
3. Inspect container health status: `docker compose ps`.
4. Check backend logs: `docker compose logs backend`.

### Re-ingestion needed (new documents or updated sources)

```bash
# Via Docker
docker compose run --rm backend python ingestion.py

# Via admin API
curl -X POST http://localhost:8000/ingest \
  -H "X-Admin-Token: your-token"
```

---

## Test priorities

When adding new features, prioritize tests in this order:

1. **Admin auth** — verify `401`/`503` on mutating routes without token; verify success with valid token.
2. **Rate limiting** — verify `429` when limit exceeded; verify success when under limit.
3. **Error contracts** — verify stable error shapes across failure modes (upstream LLM errors, DB unavailable, etc.).
