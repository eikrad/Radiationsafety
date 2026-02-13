"""API endpoint tests."""

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


def test_query_returns_answer(client: TestClient):
    """Query endpoint returns answer and sources."""
    res = client.post("/query", json={"question": "What is radiation protection?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "Test answer from mocked graph" in data["answer"]


def test_query_requires_question(client: TestClient):
    """Query with missing question returns 422."""
    res = client.post("/query", json={})
    assert res.status_code == 422


def test_query_empty_question(client: TestClient):
    """Query accepts empty string question (validation may vary)."""
    res = client.post("/query", json={"question": ""})
    assert res.status_code == 200
