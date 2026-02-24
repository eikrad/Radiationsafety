"""Pytest fixtures shared across tests."""

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env_no_api_calls(monkeypatch):
    """Prevent real API calls in tests (no LangSmith, no embeddings)."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def mock_graph():
    """Mock graph that returns a fixed result."""
    def _invoke(inputs, config=None):
        question = inputs.get("question", "")
        hist = inputs.get("chat_history") or []
        new_hist = list(hist) + [(question, "Test answer from mocked graph.")]
        return {
            "question": question,
            "generation": "Test answer from mocked graph.",
            "documents": [],
            "web_search": False,
            "web_search_attempted": False,
            "chat_history": new_hist,
            "retrieval_warning": None,
        }

    graph = MagicMock()
    graph.invoke.side_effect = _invoke
    return graph


@pytest.fixture
def client(mock_graph):
    """FastAPI test client with mocked graph."""
    from api.main import app, app_state
    app_state["graph"] = mock_graph
    with TestClient(app) as c:
        yield c
    app_state["graph"] = None
