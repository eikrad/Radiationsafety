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


def test_query_api_key_error_when_openai_selected_no_key(client: TestClient, monkeypatch):
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

    def _invoke(inputs):
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
    assert "Websuche" in data["warning"]
