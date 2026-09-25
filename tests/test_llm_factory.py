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
    assert (
        "mistral" in type(llm).__module__.lower() or "MistralAI" in type(llm).__name__
    )


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
    fake_emb = type(
        "GoogleGenerativeAIEmbeddings", (), {"__module__": "langchain_google_genai"}
    )()
    with patch(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        MagicMock(return_value=fake_emb),
    ):
        from graph.llm_factory import get_embeddings

        emb = get_embeddings()
    cls = type(emb)
    assert (
        cls.__name__ == "GoogleGenerativeAIEmbeddings"
    ), f"expected GoogleGenerativeAIEmbeddings, got {cls.__name__}"


# --- Scaleway Generative APIs (OpenAI-compatible) ---------------------------


@pytest.fixture
def scaleway_env(monkeypatch):
    monkeypatch.setenv("SCW_SECRET_KEY", "scw-test-key")
    monkeypatch.setenv("SCW_MODEL", "qwen3.8-27b")
    monkeypatch.delenv("SCW_ALLOWED_MODELS", raising=False)
    monkeypatch.delenv("SCW_BASE_URL", raising=False)


def _base_url(llm) -> str:
    return str(getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", ""))


def test_scaleway_answers_through_its_openai_compatible_endpoint(scaleway_env):
    llm = get_llm(provider="scaleway")

    assert llm.model_name == "qwen3.8-27b"
    assert _base_url(llm) == "https://api.scaleway.ai/v1"
    assert llm.temperature == 0


def test_scaleway_is_selected_by_llm_provider_env(scaleway_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "scaleway")

    assert get_llm().model_name == "qwen3.8-27b"


def test_scaleway_without_a_key_asks_for_one(scaleway_env, monkeypatch):
    monkeypatch.delenv("SCW_SECRET_KEY")

    with pytest.raises(APIKeyError, match="Scaleway"):
        get_llm(provider="scaleway")


def test_scaleway_without_a_configured_model_says_which_setting_is_missing(
    scaleway_env, monkeypatch
):
    monkeypatch.delenv("SCW_MODEL")

    with pytest.raises(ValueError, match="SCW_MODEL"):
        get_llm(provider="scaleway")


def test_a_client_cannot_pick_an_unlisted_scaleway_model(scaleway_env):
    llm = get_llm(provider="scaleway", model_variant="some/expensive-model")

    assert llm.model_name == "qwen3.8-27b"


def test_a_client_can_pick_a_scaleway_model_from_the_allow_list(
    scaleway_env, monkeypatch
):
    monkeypatch.setenv("SCW_ALLOWED_MODELS", "glm-5.2, qwen3.6-35b-a3b")

    llm = get_llm(provider="scaleway", model_variant="glm-5.2")

    assert llm.model_name == "glm-5.2"


def test_the_scaleway_endpoint_can_be_project_scoped(scaleway_env, monkeypatch):
    monkeypatch.setenv("SCW_BASE_URL", "https://api.scaleway.ai/project-123/v1")

    assert _base_url(get_llm(provider="scaleway")) == (
        "https://api.scaleway.ai/project-123/v1"
    )


def test_internal_callers_can_use_any_scaleway_model(scaleway_env):
    from graph.llm_factory import scaleway_chat

    judge = scaleway_chat("glm-5.2")

    assert judge.model_name == "glm-5.2"
    assert judge.temperature == 0


def test_scaleway_answers_still_use_gemini_embeddings():
    assert get_embedding_provider("scaleway") == "gemini"


def test_scaleway_structured_output_goes_through_tool_calling(scaleway_env):
    # json_schema-constrained decoding on Scaleway can loop until the token limit
    # (seen with glm-5.2 in grade_documents); tool calling answers reliably.
    from pydantic import BaseModel

    class Verdict(BaseModel):
        binary_score: bool

    structured = get_llm(provider="scaleway").with_structured_output(Verdict)
    request = structured.first.kwargs

    assert "response_format" not in request
    assert request["tools"][0]["function"]["name"] == "Verdict"
