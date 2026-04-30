"""API endpoint tests."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_health_without_graph():
    """Health returns graph_loaded=false when TESTING (no graph loaded)."""
    from api.main import app

    with TestClient(app) as c:
        res = c.get("/health")
    assert res.status_code == 200
    assert res.json()["graph_loaded"] is False


def test_health(client: TestClient):
    """Health endpoint returns ok when graph is loaded."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["graph_loaded"] is True


def test_metrics_returns_prometheus_style(client: TestClient):
    """Metrics endpoint returns Prometheus-style text with graph_loaded and uptime."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    text = res.text
    assert "radiationsafety_graph_loaded" in text
    assert "radiationsafety_uptime_seconds" in text
    assert "# TYPE radiationsafety_graph_loaded gauge" in text


def test_query_returns_answer(client: TestClient):
    """Query endpoint returns answer, sources, and chat_history."""
    res = client.post("/query", json={"question": "What is radiation protection?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "sources" in data
    assert "chat_history" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["chat_history"], list)
    assert "Test answer from mocked graph" in data["answer"]
    assert len(data["chat_history"]) == 1
    assert data["chat_history"][0][0] == "What is radiation protection?"


def test_query_requires_question(client: TestClient):
    """Query with missing question returns 422."""
    res = client.post("/query", json={})
    assert res.status_code == 422


def test_query_empty_question(client: TestClient):
    """Query accepts empty string question (validation may vary)."""
    res = client.post("/query", json={"question": ""})
    assert res.status_code == 200


def test_query_accepts_model_and_api_keys(client: TestClient):
    """Query accepts model and api_keys in request body."""
    res = client.post(
        "/query",
        json={
            "question": "What is radiation protection?",
            "model": "mistral",
            "api_keys": {"mistral": "test-key-from-request"},
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Test answer from mocked graph" in data["answer"]


def test_query_api_key_error_when_openai_selected_no_key(
    client: TestClient, monkeypatch
):
    """When model=openai and no API key in request or env, returns 400 with API key message."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = client.post(
        "/query",
        json={"question": "What is radiation?", "model": "openai"},
    )
    assert res.status_code == 400
    data = res.json()
    assert "detail" in data
    assert "API key" in data["detail"]


def test_query_non_question_short_circuit(client: TestClient):
    """Thank you / acknowledgments bypass graph and return friendly response."""
    res = client.post("/query", json={"question": "Thank you"})
    assert res.status_code == 200
    data = res.json()
    assert "You're welcome" in data["answer"]
    assert data["sources"] == []
    assert len(data["chat_history"]) == 1
    assert data["chat_history"][0][0] == "Thank you"


def test_query_returns_warning_when_set(client: TestClient):
    """Query endpoint returns warning when retrieval_warning is set."""
    from api.main import app, app_state

    mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

    def _invoke(inputs, config=None):
        return {
            "question": inputs.get("question", ""),
            "generation": "Answer with poor sources",
            "documents": [],
            "web_search": False,
            "web_search_attempted": True,
            "chat_history": [],
            "retrieval_warning": "Die Websuche konnte keine ausreichend guten Quellen liefern.",
        }

    mock.invoke.side_effect = _invoke
    app_state["graph"] = mock
    with TestClient(app) as c:
        res = c.post("/query", json={"question": "test"})
    assert res.status_code == 200
    data = res.json()
    assert "warning" in data
    assert data["warning"]  # warning text (language may match question)
    assert (
        "web" in data["warning"].lower()
        or "source" in data["warning"].lower()
        or "quelle" in data["warning"].lower()
    )


def test_documents_check_updates(client: TestClient):
    """Documents check-updates returns 200 and sources list."""
    mock_sources = [
        {
            "id": "test-1",
            "name": "Test Doc",
            "url": "https://example.com/doc",
            "folder": "IAEA",
            "current_version": "1.0",
            "remote_version": "2.0",
            "update_available": True,
            "local_date": None,
            "remote_date": None,
            "download_url": "https://example.com/doc2",
            "error": None,
        }
    ]
    with patch("document_updates.check_updates", return_value=mock_sources):
        res = client.get("/api/documents/check-updates")
    assert res.status_code == 200
    data = res.json()
    assert "sources" in data
    assert "recent_iaea" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == "test-1"
    assert data["sources"][0]["update_available"] is True


def test_ingest_status_idle(client: TestClient):
    """Ingest status returns idle when not running."""
    res = client.get("/api/ingest/status")
    assert res.status_code == 200
    assert res.json()["status"] == "idle"


def test_ingest_requires_admin_token(client: TestClient, monkeypatch):
    """POST ingest returns 401 without admin token."""
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    res = client.post("/api/ingest", headers={"X-Admin-Token": "wrong-token"})
    assert res.status_code == 401


def test_ingest_fails_closed_when_admin_token_not_configured(
    client: TestClient, monkeypatch
):
    """POST ingest returns 503 when ADMIN_TOKEN is not configured and bypass is off."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ADMIN_AUTH_BYPASS", "false")
    res = client.post("/api/ingest")
    assert res.status_code == 503


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/api/ingest", {}),
        ("post", "/api/documents/sync-danish", {}),
        ("post", "/api/documents/build-from-local", {}),
        ("post", "/api/documents/source/iaea-1/lookup-url", {}),
        ("post", "/api/documents/source/iaea-1/download-update", {}),
        (
            "post",
            "/api/documents/add-pdf",
            {
                "files": {
                    "file": ("test.pdf", b"%PDF-1.4 test", "application/pdf"),
                },
            },
        ),
        (
            "patch",
            "/api/documents/source/iaea-1/url",
            {
                "json": {
                    "url": "https://www.iaea.org/publications/8639/safety-assessment"
                }
            },
        ),
    ],
)
def test_mutating_routes_require_valid_admin_token(
    unauth_client: TestClient, method: str, path: str, kwargs: dict
):
    """Mutating API routes return 401 when admin token is missing/invalid."""
    request_fn = getattr(unauth_client, method)
    res = request_fn(path, headers={"X-Admin-Token": "wrong-token"}, **kwargs)
    assert res.status_code == 401


def test_public_routes_stay_accessible_without_admin_token(unauth_client: TestClient):
    """Public routes must remain available without admin token."""
    health = unauth_client.get("/health")
    assert health.status_code == 200
    query = unauth_client.post("/query", json={"question": "What is radiation safety?"})
    assert query.status_code == 200


def test_query_rate_limit_returns_429(client: TestClient, monkeypatch):
    """Query endpoint returns 429 when in-memory rate limit is exceeded."""
    from api.main import app_state

    monkeypatch.setenv("RATE_LIMIT_QUERY_MAX_REQUESTS", "1")
    monkeypatch.setenv("RATE_LIMIT_QUERY_WINDOW_SEC", "60")
    app_state["rate_limit_store"] = {}
    first = client.post("/query", json={"question": "What is radiation safety?"})
    assert first.status_code == 200
    second = client.post("/query", json={"question": "What is radiation safety?"})
    assert second.status_code == 429
    assert "retry" in second.json().get("detail", "").lower()
    assert second.headers.get("Retry-After")


def test_query_rate_limit_allows_requests_under_threshold(
    client: TestClient, monkeypatch
):
    """Query endpoint allows requests when remaining under configured threshold."""
    from api.main import app_state

    monkeypatch.setenv("RATE_LIMIT_QUERY_MAX_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_QUERY_WINDOW_SEC", "60")
    app_state["rate_limit_store"] = {}
    for _ in range(3):
        res = client.post("/query", json={"question": "What is radiation safety?"})
        assert res.status_code == 200


def test_admin_route_rate_limit_returns_429(client: TestClient, monkeypatch):
    """Admin route returns 429 when admin rate limit is exceeded."""
    from api.main import app_state

    monkeypatch.setenv("RATE_LIMIT_ADMIN_MAX_REQUESTS", "1")
    monkeypatch.setenv("RATE_LIMIT_ADMIN_WINDOW_SEC", "60")
    app_state["rate_limit_store"] = {}
    with patch("api.main._run_ingest"):
        first = client.post("/api/ingest")
        assert first.status_code == 202
        second = client.post("/api/ingest")
        assert second.status_code == 429
        assert second.headers.get("Retry-After")


def test_query_rate_limit_redis_backend(client: TestClient, monkeypatch):
    """Query rate limiting works with optional Redis backend."""
    from api.main import app_state

    class _FakeRedis:
        def __init__(self):
            self.counts: dict[str, int] = {}

        def incr(self, key: str) -> int:
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

        def expire(self, key: str, _seconds: int) -> bool:
            return True

    monkeypatch.setenv("RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setenv("RATE_LIMIT_REDIS_URL", "redis://example.test:6379/0")
    monkeypatch.setenv("RATE_LIMIT_QUERY_MAX_REQUESTS", "1")
    monkeypatch.setenv("RATE_LIMIT_QUERY_WINDOW_SEC", "60")
    app_state["rate_limit_store"] = {}
    app_state["rate_limit_redis_client"] = _FakeRedis()

    first = client.post("/query", json={"question": "What is radiation safety?"})
    assert first.status_code == 200
    second = client.post("/query", json={"question": "What is radiation safety?"})
    assert second.status_code == 429
    assert second.headers.get("Retry-After")


def test_query_sets_request_id_header(client: TestClient):
    """Query responses include an X-Request-ID header for correlation."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID")


def test_request_id_header_is_preserved_when_provided(client: TestClient):
    """Server preserves inbound X-Request-ID for easier cross-service tracing."""
    request_id = "req-test-1234"
    res = client.get("/health", headers={"X-Request-ID": request_id})
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == request_id


def test_metrics_include_http_counters(client: TestClient, monkeypatch):
    """Metrics include request totals and error totals after traffic."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ADMIN_AUTH_BYPASS", "false")
    client.get("/health")
    client.post("/api/ingest")
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "radiationsafety_http_requests_total" in res.text
    assert "radiationsafety_http_errors_total" in res.text


def test_metrics_include_endpoint_level_counters(client: TestClient, monkeypatch):
    """Metrics expose per-endpoint request/error counters."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ADMIN_AUTH_BYPASS", "false")
    client.get("/health")
    client.post("/api/ingest")
    res = client.get("/metrics")
    assert res.status_code == 200
    assert 'radiationsafety_http_requests_by_path_total{path="/health"}' in res.text
    assert 'radiationsafety_http_errors_by_path_total{path="/api/ingest"}' in res.text


def test_metrics_include_status_class_counters(client: TestClient, monkeypatch):
    """Metrics expose counters by HTTP status code class."""
    monkeypatch.setenv("RATE_LIMIT_QUERY_MAX_REQUESTS", "1")
    monkeypatch.setenv("RATE_LIMIT_QUERY_WINDOW_SEC", "60")
    from api.main import app_state

    app_state["rate_limit_store"] = {}
    client.get("/health")
    client.post("/query", json={"question": "What is radiation safety?"})
    client.post("/query", json={"question": "What is radiation safety?"})
    res = client.get("/metrics")
    assert res.status_code == 200
    assert (
        'radiationsafety_http_responses_by_status_class_total{status_class="2xx"}'
        in res.text
    )
    assert (
        'radiationsafety_http_responses_by_status_class_total{status_class="4xx"}'
        in res.text
    )


def test_metrics_count_query_web_search_attempts(client: TestClient):
    """Metrics increase web-search-attempts counter when query pipeline reports fallback."""
    from api.main import app_state

    app_state["request_metrics"]["query_web_search_attempts"] = 0
    mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

    def _invoke(inputs, config=None):
        question = inputs.get("question", "")
        return {
            "question": question,
            "generation": "Answer from web fallback",
            "documents": [],
            "web_search": False,
            "web_search_attempted": True,
            "chat_history": [(question, "Answer from web fallback")],
            "retrieval_warning": None,
        }

    mock.invoke.side_effect = _invoke
    app_state["graph"] = mock
    client.post("/query", json={"question": "Need latest guidance"})
    res = client.get("/metrics")
    assert res.status_code == 200
    value = 0
    for line in res.text.splitlines():
        if line.startswith("radiationsafety_query_web_search_attempts_total "):
            value = int(line.rsplit(" ", 1)[-1])
            break
    assert value >= 1


def test_metrics_include_query_outcome_labels(client: TestClient):
    """Metrics expose query outcome labels from graph routing result."""
    from api.main import app_state

    app_state["request_metrics"]["query_outcomes_total"] = {}
    mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

    def _invoke(inputs, config=None):
        question = inputs.get("question", "")
        return {
            "question": question,
            "generation": "Answer",
            "documents": [],
            "web_search": False,
            "web_search_attempted": False,
            "chat_history": [(question, "Answer")],
            "retrieval_warning": None,
            "routing_outcome": "trusted_only_verified",
            "trusted_verified": True,
        }

    mock.invoke.side_effect = _invoke
    app_state["graph"] = mock
    client.post("/query", json={"question": "Need guidance"})
    res = client.get("/metrics")
    assert res.status_code == 200
    assert (
        'radiationsafety_query_outcomes_total{outcome="trusted_only_verified"} 1'
        in res.text
    )


def test_ingest_returns_accepted(client: TestClient):
    """POST ingest returns 202 and starts background task."""
    with patch("api.main._run_ingest"):
        res = client.post("/api/ingest")
    assert res.status_code == 202
    data = res.json()
    assert data.get("status") == "accepted"


def test_documents_set_source_url_rejects_disallowed_url(client: TestClient):
    """PATCH source URL returns 400 when URL is not from iaea.org or retsinformation.dk."""
    res = client.patch(
        "/api/documents/source/iaea-1/url",
        json={"url": "https://evil.com/doc"},
    )
    assert res.status_code == 400
    detail = res.json().get("detail", "")
    assert "retsinformation" in detail or "iaea" in detail.lower()


def test_documents_set_source_url_not_found(client: TestClient):
    """PATCH source URL returns 404 when source_id is not in registry."""
    with patch("document_updates.load_registry_raw", return_value=[]):
        res = client.patch(
            "/api/documents/source/nonexistent/url",
            json={"url": "https://www.iaea.org/publications/123/test"},
        )
    assert res.status_code == 404


def test_documents_set_source_url_ok(client: TestClient):
    """PATCH source URL updates registry when URL is allowlisted and source exists."""
    with patch(
        "document_updates.load_registry_raw",
        return_value=[{"id": "iaea-1", "name": "Doc"}],
    ):
        with patch("document_updates._allowed_url", return_value=True):
            with patch("document_updates.update_registry_url") as upd:
                res = client.patch(
                    "/api/documents/source/iaea-1/url",
                    json={
                        "url": "https://www.iaea.org/publications/8639/safety-assessment"
                    },
                )
    assert res.status_code == 200
    assert res.json().get("ok") is True
    upd.assert_called_once_with(
        "iaea-1", "https://www.iaea.org/publications/8639/safety-assessment"
    )


def test_documents_get_source_file_not_found(client: TestClient):
    """GET source file returns 404 when source_id is not in registry."""
    with patch("document_updates._load_registry", return_value=[]):
        res = client.get("/api/documents/source/nonexistent/file")
    assert res.status_code == 404


def test_documents_get_source_file_no_local_file(client: TestClient):
    """GET source file returns 404 when source exists but has no local PDF."""
    from document_updates import DocumentSource

    with patch(
        "document_updates._load_registry",
        return_value=[
            DocumentSource(
                id="iaea-1",
                name="Doc",
                url="https://iaea.org/x",
                folder="IAEA",
                filename_hint=None,
            ),
        ],
    ):
        with patch("document_updates.get_local_pdf_path", return_value=None):
            res = client.get("/api/documents/source/iaea-1/file")
    assert res.status_code == 404


def test_documents_get_source_file_ok(client: TestClient, tmp_path: Path):
    """GET source file returns 200 with PDF when local file exists."""
    from document_updates import DocumentSource

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")
    source = DocumentSource(
        id="iaea-1",
        name="Doc",
        url="https://iaea.org/x",
        folder="IAEA",
        filename_hint=None,
    )
    with patch("document_updates._load_registry", return_value=[source]):
        with patch("document_updates.get_local_pdf_path", return_value=pdf):
            res = client.get("/api/documents/source/iaea-1/file")
    assert res.status_code == 200
    assert res.headers.get("content-type", "").startswith("application/pdf")
    assert res.content == b"%PDF-1.4 minimal"


def test_documents_sync_danish(client: TestClient):
    """Sync endpoint returns report from document_updates service."""
    report = {
        "mode": "shadow",
        "apply_updates": False,
        "updated_count": 0,
        "checked_count": 1,
        "items": [{"id": "dk-1", "status": "unchanged"}],
    }
    with patch("document_updates.sync_danish_legislation", return_value=report):
        res = client.post("/api/documents/sync-danish")
    assert res.status_code == 200
    assert res.json()["mode"] == "shadow"
    assert res.json()["checked_count"] == 1
