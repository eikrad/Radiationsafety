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
| `RATE_LIMIT_ADMIN_MAX_REQUESTS` | `10` | Max admin requests per window |
| `RATE_LIMIT_ADMIN_WINDOW_SEC` | `60` | Admin window size in seconds |

**Single-process deployments:** `in_memory` is suitable. Limits are per-client (IP-based).

**Multi-worker / multi-replica deployments:** set `RATE_LIMIT_BACKEND=redis` and provide `RATE_LIMIT_REDIS_URL`. Without this, each worker has its own counter and limits are not enforced globally.

---

## Observability

- Every HTTP response includes an `X-Request-ID` header for log correlation.
- `GET /metrics` exports Prometheus-style counters:
  - `requests_total` — total requests by route
  - `errors_total` — total errors by route
  - `request_duration_seconds_sum` — cumulative request durations
  - `query_web_search_total` — queries that triggered web search fallback

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
