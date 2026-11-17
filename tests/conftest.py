"""Pytest fixtures shared across tests."""

import dotenv
import pytest
from fastapi.testclient import TestClient

# A developer's .env must not reach the tests. Modules call load_dotenv() when
# first imported, which can happen inside a test after the fixture below has
# cleared the environment (ingestion.py brought OLLAMA_EMBED_MODEL back).
dotenv.load_dotenv = lambda *args, **kwargs: False


@pytest.fixture(autouse=True)
def _env_no_api_calls(monkeypatch):
    """Prevent real API calls in tests (no LangSmith, no embeddings)."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SCW_SECRET_KEY", "test-key")
    monkeypatch.setenv("SCW_MODEL", "gemma-4-26b-a4b-it")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    # A developer's .env (loaded at import) must not change what the tests see:
    # the defaults are what CI and a fresh checkout get.
    for name in (
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "SCW_ALLOWED_MODELS",
        "SCW_BASE_URL",
        "EMBED_QUERY_INSTRUCTION",
        "EVAL_GRADER_PROVIDER",
        "EVAL_JUDGE_MODEL",
        "EVAL_JUDGE_VOTES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    monkeypatch.delenv("ADMIN_AUTH_BYPASS", raising=False)


@pytest.fixture
def stub_answerer():
    """The Answerer behind the API, replaced by a stub (tests.fakes.StubAnswerer)."""
    from tests.fakes import StubAnswerer

    return StubAnswerer()


@pytest.fixture
def client(stub_answerer):
    """FastAPI test client with a stub answerer."""
    from api.main import app, app_state

    app_state["answerer"] = stub_answerer
    with TestClient(app) as c:
        c.headers.update({"X-Admin-Token": "test-admin-token"})
        yield c
    app_state["answerer"] = None


@pytest.fixture
def unauth_client(stub_answerer):
    """FastAPI test client with a stub answerer and no admin token."""
    from api.main import app, app_state

    app_state["answerer"] = stub_answerer
    with TestClient(app) as c:
        yield c
    app_state["answerer"] = None
