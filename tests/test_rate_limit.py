"""Tests for api/rate_limit.py: proxy-header trust and in-memory store cleanup."""

from starlette.requests import Request

from api.rate_limit import _sweep_stale_entries, request_client_key


def _make_request(
    *, client_host: str | None = "203.0.113.5", xff: str | None = None
) -> Request:
    """Build a minimal Starlette Request with an optional client and X-Forwarded-For header."""
    headers = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode()))
    scope = {
        "type": "http",
        "headers": headers,
        "client": (client_host, 12345) if client_host else None,
    }
    return Request(scope)


class TestRequestClientKey:
    """X-Forwarded-For must only be trusted when explicitly enabled."""

    def test_ignores_xff_by_default(self, monkeypatch):
        """Without TRUST_PROXY_HEADERS, X-Forwarded-For is ignored even if present."""
        monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
        request = _make_request(client_host="203.0.113.5", xff="198.51.100.1")
        assert request_client_key(request) == "203.0.113.5"

    def test_uses_xff_when_trust_enabled(self, monkeypatch):
        """With TRUST_PROXY_HEADERS=true, the first X-Forwarded-For hop is used."""
        monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
        request = _make_request(client_host="203.0.113.5", xff="198.51.100.1, 10.0.0.1")
        assert request_client_key(request) == "198.51.100.1"

    def test_falls_back_to_client_host_when_no_xff(self, monkeypatch):
        """With TRUST_PROXY_HEADERS=true but no header present, falls back to client.host."""
        monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
        request = _make_request(client_host="203.0.113.5", xff=None)
        assert request_client_key(request) == "203.0.113.5"

    def test_unknown_when_no_client_at_all(self, monkeypatch):
        """No client and no trusted XFF returns 'unknown' rather than crashing."""
        monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
        request = _make_request(client_host=None, xff="198.51.100.1")
        assert request_client_key(request) == "unknown"


class TestSweepStaleEntries:
    """In-memory store must not grow forever for a long-running process."""

    def test_removes_entries_older_than_ttl(self):
        now = 10_000.0
        store = {
            "query:1.2.3.4": (now - 4000.0, 3),  # stale (> 3600s old)
            "query:5.6.7.8": (now - 10.0, 1),  # fresh
        }
        app_state: dict = {}
        _sweep_stale_entries(store, app_state, now)
        assert "query:1.2.3.4" not in store
        assert "query:5.6.7.8" in store

    def test_does_not_sweep_more_than_once_per_interval(self):
        now = 10_000.0
        store = {"query:1.2.3.4": (now - 4000.0, 3)}
        app_state: dict = {"rate_limit_last_sweep": now - 10.0}  # swept 10s ago
        _sweep_stale_entries(store, app_state, now)
        # Interval (300s) hasn't elapsed since the last sweep, so the stale
        # entry is left in place even though it's past the TTL.
        assert "query:1.2.3.4" in store

    def test_sweeps_again_after_interval_elapses(self):
        now = 10_000.0
        store = {"query:1.2.3.4": (now - 4000.0, 3)}
        app_state: dict = {"rate_limit_last_sweep": now - 400.0}  # > 300s ago
        _sweep_stale_entries(store, app_state, now)
        assert "query:1.2.3.4" not in store
        assert app_state["rate_limit_last_sweep"] == now
