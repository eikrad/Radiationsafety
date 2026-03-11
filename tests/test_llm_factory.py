"""LLM and embeddings factory tests."""

from unittest.mock import MagicMock, patch

import pytest

from graph.llm_factory import APIKeyError, get_embedding_provider, get_llm


def test_get_llm_returns_gemini_by_default(monkeypatch):
    """Default provider is gemini (2.5 Pro) when LLM_PROVIDER not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    llm = get_llm()
    assert "google" in type(llm).__module__.lower() or "Google" in type(llm).__name__
    assert getattr(llm, "model", None) == "gemini-2.5-pro"


def test_get_llm_returns_mistral_when_set(monkeypatch):
    """When LLM_PROVIDER=mistral, returns ChatMistralAI."""
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    llm = get_llm()
    assert "mistral" in type(llm).__module__.lower() or "MistralAI" in type(llm).__name__


def test_get_llm_gemini_uses_GEMINI_MODEL_env(monkeypatch):
    """When LLM_PROVIDER=gemini and GEMINI_MODEL is set, that model is used."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    llm = get_llm()
    assert getattr(llm, "model", None) == "gemini-2.5-pro"


def test_get_llm_returns_openai_when_set(monkeypatch):
    """When provider=openai, returns ChatOpenAI."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = get_llm(provider="openai")
    assert "openai" in type(llm).__module__.lower() or "OpenAI" in type(llm).__name__


def test_get_llm_raises_api_key_error_when_openai_key_missing(monkeypatch):
    """When provider=openai and no API key, raises APIKeyError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(APIKeyError) as exc_info:
        get_llm(provider="openai")
    assert "OpenAI" in str(exc_info.value)


def test_get_embedding_provider_gemini_and_openai_use_gemini(monkeypatch):
    """Gemini and OpenAI LLM providers both use 'gemini' embedding provider."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert get_embedding_provider() == "gemini"
    assert get_embedding_provider("openai") == "gemini"
    assert get_embedding_provider("gemini") == "gemini"


def test_get_embedding_provider_mistral_uses_gemini_embeddings(monkeypatch):
    """Mistral LLM uses 'gemini' embedding provider so it can use the shared vector store."""
    assert get_embedding_provider("mistral") == "gemini"


def test_get_embeddings_returns_gemini_by_default(monkeypatch):
    """Default embeddings are Google when LLM_PROVIDER not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    fake_emb = type("GoogleGenerativeAIEmbeddings", (), {"__module__": "langchain_google_genai"})()
    with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", MagicMock(return_value=fake_emb)):
        from graph.llm_factory import get_embeddings

        emb = get_embeddings()
    cls = type(emb)
    assert cls.__name__ == "GoogleGenerativeAIEmbeddings", f"expected GoogleGenerativeAIEmbeddings, got {cls.__name__}"
