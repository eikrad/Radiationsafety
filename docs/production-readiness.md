# Production Readiness Reference

Operational reference for the Radiation Safety RAG API. Covers security defaults, rate limiting, observability, container hardening, and runbook quick checks.

---

## Route Exposure Matrix

| Route | Auth required | Notes |
|---|---|---|
| `GET /health` | No | Returns `{"status": "ok", "graph_loaded": bool}` |
| `GET /metrics` | No | Prometheus-format text export |
| `GET /config` | No | Returns `{"server_has_llm_key": bool}` (UI hint) |
| `POST /query` | No | Main RAG endpoint; rate-limited |
| `GET /documents/check-updates` | No | Lists registered sources with version info |
| `GET /documents/source/{id}/file` | No | Serves local PDF |
| `POST /ingest` | **Yes** | Triggers background ingestion |
| `GET /ingest/status` | No | Returns ingestion state (idle/running) |
| `PATCH /documents/source/{id}/url` | **Yes** | Update document URL in registry |
| `POST /documents/source/{id}/lookup-url` | **Yes** | Web-search for document URL |
| `POST /documents/source/{id}/download-update` | **Yes** | Download new version, backup old |
| `POST /documents/sync-danish` | **Yes** | Sync Danish legislation via Harvest + ELI |
| `POST /documents/build-from-local` | **Yes** | Build registry from local PDFs |
| `POST /documents/add-pdf` | **Yes** | Upload PDF, add to collection, append to registry |

---

## Admin Authentication

- All admin routes require the `X-Admin-Token` request header matching the `ADMIN_TOKEN` env var.
- **Fail-closed default**: if `ADMIN_TOKEN` is not set, admin routes return `503 Service Unavailable`. This prevents accidental open access.
- `secrets.compare_digest()` is used for constant-time comparison, preventing timing attacks.
- **Local bypass**: `ADMIN_AUTH_BYPASS=true` skips the token check. Intended for local development only — never set this in production.

---

## Rate Limiting

- Default backend: in-memory, per client (IP-based), suitable for single-process deployments.
- Multi-worker/replica deployments: set `RATE_LIMIT_BACKEND=redis` and `RATE_LIMIT_REDIS_URL` to enforce global limits across processes.

| Env var | Default | Description |
|---|---|---|
| `RATE_LIMIT_BACKEND` | `in_memory` | `in_memory` or `redis` |
| `RATE_LIMIT_REDIS_URL` | — | e.g. `redis://localhost:6379/0` |
| `RATE_LIMIT_QUERY_MAX_REQUESTS` | `60` | Max `/query` calls per window |
| `RATE_LIMIT_QUERY_WINDOW_SEC` | `60` | Window size in seconds |
| `RATE_LIMIT_ADMIN_MAX_REQUESTS` | `20` | Max admin calls per window |
| `RATE_LIMIT_ADMIN_WINDOW_SEC` | `60` | Admin window size in seconds |

Rate-limit violations return `429 Too Many Requests`.

---

## Observability

### Request tracing

Every HTTP response includes an `X-Request-ID` header for correlation across logs and client errors.

### Metrics endpoint (`GET /metrics`)

Returns Prometheus-format text. Exported metrics:

| Metric | Description |
|---|---|
| `http_requests_total` | Total requests, labelled by path and status class (`2xx`, `4xx`, `5xx`) |
| `http_errors_total` | Total error responses |
| `http_request_duration_seconds_sum` | Cumulative request duration in seconds |
| `web_search_attempts_total` | Number of times web search was triggered |
| `query_outcome_total` | Per-outcome counts: `trusted_only_verified`, `trusted_only_unverified`, `web_search_verified`, `web_search_unverified` |

### Health check (`GET /health`)

Returns `{"status": "ok", "graph_loaded": true}`. Use this for liveness and readiness probes.

---

## Container Hardening

The Docker setup applies defence-in-depth at the container level:

- Backend runs as non-root user (`appuser`).
- `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1` are set in the image.
- Compose applies `no-new-privileges: true` and `cap_drop: [ALL]` for both backend and frontend services.
- Backend uses `tmpfs` for `/tmp` and a persistent named volume only for `/app/.chroma` (the vector DB).
- Frontend container does not start until the backend passes its health check.

---

## Runbook Quick Checks

**429 spike (rate limit errors)**
- Check `RATE_LIMIT_QUERY_MAX_REQUESTS` and `RATE_LIMIT_QUERY_WINDOW_SEC` against recent traffic.
- Inspect `/metrics` for `http_requests_total` by path to identify the source.
- If running multi-worker, verify `RATE_LIMIT_BACKEND=redis` is set and the Redis instance is reachable.

**Admin routes returning 503**
- Check that `ADMIN_TOKEN` is set in the environment.
- Verify `ADMIN_AUTH_BYPASS` is not accidentally set to `true` in non-local environments.

**Service health degraded**
- Call `GET /health` — check `graph_loaded` field.
- Inspect `/metrics` for error rate and duration spikes.
- Check container health status in `docker compose ps`.

**Ingestion taking too long or failing**
- `GET /ingest/status` shows current state (`idle` / `running`).
- Free-tier Gemini embeddings require `GEMINI_BATCH_DELAY_SEC=65` between batches — normal for large document sets.
- Check logs for rate-limit errors from the Gemini embedding API.

---

## Test Priorities

Ordered by risk and blast radius:

1. Admin auth behavior on all mutating routes — `503` when no token set, `401` on wrong token, `200` on correct token.
2. Rate-limit behavior — `429` after limit exceeded, `200` within limit.
3. `/query` response contract — shape of `QueryResponse` (answer, sources, chat_history, warning fields).
4. Non-question short-circuit — greetings must not invoke the graph.
5. Failure modes — graph errors surface as `500` with a stable error body; no raw stack traces in responses.
