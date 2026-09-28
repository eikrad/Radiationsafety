"""GET /config reports, per provider, whether this server can answer with it."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _build_search_index(chroma_dir: Path, embedding_provider: str) -> None:
    """Create both collections for `embedding_provider` with one chunk each."""
    import chromadb

    from ingestion import get_collection_names

    client = chromadb.PersistentClient(path=str(chroma_dir))
    for name in get_collection_names(embedding_provider):
        client.get_or_create_collection(name).add(
            ids=["chunk-1"], documents=["ALARA"], embeddings=[[0.1, 0.2, 0.3]]
        )


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch) -> Path:
    """An empty Chroma store in place of the project's .chroma."""
    monkeypatch.setattr("ingestion._CHROMA_DIR", tmp_path)
    return tmp_path


def test_scaleway_without_an_answer_model_needs_setup(
    client: TestClient, chroma_dir, monkeypatch
):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    _build_search_index(chroma_dir, "gemini")
    monkeypatch.delenv("SCW_MODEL", raising=False)

    providers = client.get("/config").json()["providers"]

    assert "SCW_MODEL" in providers["scaleway"]["issue"]
    assert providers["gemini"]["issue"] is None


def test_a_provider_whose_search_index_is_not_built_needs_setup(
    client: TestClient, chroma_dir, monkeypatch
):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "scaleway")
    _build_search_index(chroma_dir, "gemini")

    providers = client.get("/config").json()["providers"]

    assert "not built" in providers["scaleway"]["issue"]
    assert "not built" in providers["gemini"]["issue"]


def test_search_embeddings_need_the_server_key_even_with_a_browser_key(
    client: TestClient, chroma_dir, monkeypatch
):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    _build_search_index(chroma_dir, "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY")

    providers = client.get("/config").json()["providers"]

    assert "GOOGLE_API_KEY" in providers["openai"]["issue"]


def test_ollama_searches_its_own_local_index(
    client: TestClient, chroma_dir, monkeypatch
):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    _build_search_index(chroma_dir, "gemini")

    providers = client.get("/config").json()["providers"]

    assert providers["gemini"]["issue"] is None
    assert "Local embeddings are not built" in providers["ollama"]["issue"]


def test_reports_which_answer_keys_the_server_holds(
    client: TestClient, chroma_dir, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY")

    providers = client.get("/config").json()["providers"]

    assert providers["openai"]["server_key"] is False
    assert providers["gemini"]["server_key"] is True
    assert providers["ollama"]["server_key"] is False
