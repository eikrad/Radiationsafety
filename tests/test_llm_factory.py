"""LLM and embeddings factory tests."""

from unittest.mock import MagicMock, patch

import pytest

from graph.llm_factory import APIKeyError, get_llm


def test_get_llm_returns_mistral_by_default(monkeypatch):
    """Default provider is mistral when LLM_PROVIDER not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = get_llm()
    assert "mistral" in type(llm).__module__.lower() or "MistralAI" in type(llm).__name__


def test_get_llm_returns_gemini_when_set(monkeypatch):
    """When LLM_PROVIDER=gemini, returns ChatGoogleGenerativeAI."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    llm = get_llm()
    assert "google" in type(llm).__module__.lower() or "Google" in type(llm).__name__


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


def test_get_embeddings_returns_mistral_by_default(monkeypatch):
    """Default embeddings are MistralAI when LLM_PROVIDER not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    # MistralAIEmbeddings downloads a HuggingFace tokenizer on init; mock to avoid network.
    fake_emb = type("MistralAIEmbeddings", (), {"__module__": "langchain_mistralai.embeddings"})()
    with patch("langchain_mistralai.MistralAIEmbeddings", MagicMock(return_value=fake_emb)):
        from graph.llm_factory import get_embeddings

        emb = get_embeddings()
    cls = type(emb)
    assert cls.__name__ == "MistralAIEmbeddings", f"expected MistralAIEmbeddings, got {cls.__name__}"
    assert cls.__module__.startswith("langchain_mistralai"), (
        f"expected langchain_mistralai.*, got {cls.__module__}"
    )
