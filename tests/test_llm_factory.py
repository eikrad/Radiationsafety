"""LLM and embeddings factory tests."""

import pytest


def test_get_llm_returns_mistral_by_default(monkeypatch):
    """Default provider is mistral when LLM_PROVIDER not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from graph.llm_factory import get_llm

    llm = get_llm()
    assert "mistral" in type(llm).__module__.lower() or "MistralAI" in type(llm).__name__


def test_get_llm_returns_gemini_when_set(monkeypatch):
    """When LLM_PROVIDER=gemini, returns ChatGoogleGenerativeAI."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    from graph.llm_factory import get_llm

    llm = get_llm()
    assert "google" in type(llm).__module__.lower() or "Google" in type(llm).__name__


def test_get_embeddings_returns_mistral_by_default(monkeypatch):
    """Default embeddings are MistralAI when LLM_PROVIDER not set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from graph.llm_factory import get_embeddings

    emb = get_embeddings()
    assert "mistral" in type(emb).__module__.lower() or "Mistral" in type(emb).__name__
