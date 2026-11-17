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


_CHUNKS = {
    "iaea": [
        ("III.1. The occupational dose limit is 20 mSv per year.", "GSR-3.pdf", "IAEA"),
        ("III.2. Averaged over five years, 100 mSv.", "GSR-3.pdf", "IAEA"),
    ],
    "dk": [
        (
            "§ 14. Dosisgrænsen for arbejdstagere er 20 mSv pr. år.",
            "BEK-2025-138405",
            "Danish law",
        ),
    ],
}


@pytest.fixture
def temp_search_index(tmp_path, monkeypatch):
    """A search index in tmp_path for the cloud and the local embeddings;
    returns the embeddings, which note the tracing state at every call."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from graph import search_index
    from tests.fakes import RecordingEmbeddings

    recording = RecordingEmbeddings()
    monkeypatch.setattr(search_index, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(search_index, "get_embeddings", lambda ep=None: recording)
    search_index.refresh()
    for provider in ("scaleway", "ollama"):
        for name, chunks in zip(
            search_index.collection_names(provider),
            (_CHUNKS["iaea"], _CHUNKS["dk"]),
            strict=True,
        ):
            docs = [
                Document(
                    page_content=text, metadata={"source": src, "document_type": t}
                )
                for text, src, t in chunks
            ]
            Chroma.from_documents(
                docs, recording, collection_name=name, persist_directory=str(tmp_path)
            )
    recording.tracing_seen.clear()
    yield recording
    search_index.refresh()


@pytest.fixture
def web_queries(monkeypatch):
    """Every query web search sends; the web itself is never reached."""
    import importlib

    queries: list[str] = []

    class RecordingBrave:
        @classmethod
        def from_api_key(cls, api_key, search_kwargs=None):
            return cls()

        def invoke(self, query):
            queries.append(query)
            return "[]"

    # the module, not the node function graph.nodes re-exports under its name
    module = importlib.import_module("graph.nodes.web_search")
    monkeypatch.setattr(module, "BraveSearch", RecordingBrave)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    return queries
