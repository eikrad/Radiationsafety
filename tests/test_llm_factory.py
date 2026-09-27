"""LLM and embeddings factory tests."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from graph.llm_factory import APIKeyError, get_embedding_provider, get_llm


def test_get_llm_returns_gemini_2_5_pro_when_chosen(monkeypatch):
    """LLM_PROVIDER=gemini answers with Gemini 2.5 Pro unless GEMINI_MODEL says otherwise."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
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


def test_chosen_gemini_embeddings_serve_every_cloud_answer_model(monkeypatch):
    """EMBEDDING_PROVIDER=gemini: Gemini and OpenAI answers both retrieve with Gemini."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    assert get_embedding_provider() == "gemini"
    assert get_embedding_provider("openai") == "gemini"
    assert get_embedding_provider("gemini") == "gemini"


def test_chosen_gemini_embeddings_serve_mistral_answers(monkeypatch):
    """Mistral answers retrieve from the shared vector store of the chosen embeddings."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    assert get_embedding_provider("mistral") == "gemini"


def test_get_embeddings_returns_gemini_when_chosen(monkeypatch):
    """EMBEDDING_PROVIDER=gemini returns Google embeddings."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
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


class _Verdict(BaseModel):
    passed: bool
    missing_info: str = ""


@pytest.fixture
def scaleway_replies(scaleway_env, monkeypatch):
    """Answer Scaleway chat requests with a canned reply; keep the request bodies."""
    import httpx

    sent: list[dict] = []
    reply = {}

    def send(self, request, **_):
        sent.append(json.loads(request.content))
        message = {"role": "assistant", "content": reply.get("content")}
        if "tool_args" in reply:
            message["tool_calls"] = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "_Verdict",
                        "arguments": json.dumps(reply["tool_args"]),
                    },
                }
            ]
        body = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3.8-27b",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return SimpleNamespace(sent=sent, reply=reply)


def _ask_for_verdict():
    return get_llm(provider="scaleway").with_structured_output(_Verdict).invoke("ok?")


def test_scaleway_structured_output_goes_through_tool_calling(scaleway_replies):
    # json_schema-constrained decoding on Scaleway can loop until the token limit
    # (seen with glm-5.2 in grade_documents); tool calling answers reliably.
    scaleway_replies.reply["tool_args"] = {"passed": False, "missing_info": "annex 2"}

    assert _ask_for_verdict() == _Verdict(passed=False, missing_info="annex 2")
    [request] = scaleway_replies.sent
    assert "response_format" not in request
    assert request["tools"][0]["function"]["name"] == "_Verdict"


def test_a_structured_reply_written_as_text_is_still_understood(scaleway_replies):
    # gemma-4 on Scaleway sometimes ignores the forced tool call and writes the
    # JSON as a fenced block instead
    scaleway_replies.reply["content"] = (
        '```json\n{\n "passed": true,\n "missing_info": ""\n}\n```'
    )

    assert _ask_for_verdict() == _Verdict(passed=True)


def test_an_unreadable_structured_reply_fails_loudly_instead_of_returning_none(
    scaleway_replies,
):
    scaleway_replies.reply["content"] = "I think the answer is fine."

    with pytest.raises(ValueError, match="_Verdict"):
        _ask_for_verdict()


# --- Embeddings chosen separately from the answer model ----------------------


@pytest.fixture
def scaleway_embedding_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "scaleway")
    monkeypatch.setenv("SCW_SECRET_KEY", "scw-test-key")
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")
    monkeypatch.delenv("EMBED_QUERY_INSTRUCTION", raising=False)
    monkeypatch.delenv("SCW_BASE_URL", raising=False)


def test_the_embedding_provider_is_chosen_independently_of_the_answer_model(
    scaleway_embedding_env,
):
    assert get_embedding_provider("gemini") == "scaleway"
    assert get_embedding_provider("scaleway") == "scaleway"
    assert get_embedding_provider("mistral") == "scaleway"


def test_privacy_mode_keeps_local_embeddings_whatever_is_configured(
    scaleway_embedding_env,
):
    assert get_embedding_provider("ollama") == "ollama"


def test_an_unknown_embedding_provider_names_the_valid_ones(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cohere")

    with pytest.raises(ValueError, match="gemini, ollama, scaleway"):
        get_embedding_provider("gemini")


@pytest.fixture
def embedding_requests(scaleway_embedding_env, monkeypatch):
    """Answer Scaleway embedding requests with small vectors; keep the request bodies."""
    import base64
    import struct

    import httpx

    sent: list[dict] = []

    def send(self, request, **_):
        body = json.loads(request.content)
        sent.append({"url": str(request.url), **body})
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        vector = [0.25, 0.5, 0.75]
        if body.get("encoding_format") == "base64":
            vector = base64.b64encode(struct.pack("<3f", *vector)).decode()
        data = [
            {"object": "embedding", "index": i, "embedding": vector}
            for i in range(len(inputs))
        ]
        reply = {"object": "list", "data": data, "model": body["model"], "usage": {}}
        return httpx.Response(200, json=reply, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return sent


def _embedded_texts(request: dict) -> list:
    return (
        request["input"] if isinstance(request["input"], list) else [request["input"]]
    )


def test_questions_carry_the_models_instruction_and_documents_stay_plain(
    embedding_requests,
):
    from graph.llm_factory import get_embeddings

    embeddings = get_embeddings("scaleway")
    embeddings.embed_documents(["Dosisgrænserne fremgår af bilag 2."])
    embeddings.embed_query("Hvor findes dosisgrænserne?")

    documents, query = embedding_requests
    assert documents["url"].startswith("https://api.scaleway.ai/v1/")
    assert documents["model"] == "qwen3-embedding-8b"
    # raw text, not tiktoken ids: Scaleway embeds strings
    assert _embedded_texts(documents) == ["Dosisgrænserne fremgår af bilag 2."]
    [question] = _embedded_texts(query)
    assert question.startswith("Instruct: ")
    assert question.endswith("\nQuery:Hvor findes dosisgrænserne?")


def test_bge_gets_its_own_instruction_format(embedding_requests, monkeypatch):
    from graph.llm_factory import get_embeddings

    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    get_embeddings("scaleway").embed_query("Hvor findes dosisgrænserne?")

    [question] = _embedded_texts(embedding_requests[0])
    assert question.startswith("<instruct>")
    assert question.endswith("\n<query>Hvor findes dosisgrænserne?")


def test_the_query_instruction_can_be_switched_off(embedding_requests, monkeypatch):
    from graph.llm_factory import get_embeddings

    monkeypatch.setenv("EMBED_QUERY_INSTRUCTION", "false")
    get_embeddings("scaleway").embed_query("Hvor findes dosisgrænserne?")

    assert _embedded_texts(embedding_requests[0]) == ["Hvor findes dosisgrænserne?"]


def test_scaleway_embeddings_need_a_configured_model(
    scaleway_embedding_env, monkeypatch
):
    from graph.llm_factory import get_embedding_model_name, get_embeddings

    assert get_embedding_model_name("scaleway") == "qwen3-embedding-8b"
    monkeypatch.delenv("SCW_EMBED_MODEL")
    with pytest.raises(ValueError, match="SCW_EMBED_MODEL"):
        get_embeddings("scaleway")


# --- Scaleway is the default provider -----------------------------------------


@pytest.fixture
def nothing_configured(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "SCW_ALLOWED_MODELS",
        "SCW_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SCW_SECRET_KEY", "scw-test-key")
    monkeypatch.setenv("SCW_MODEL", "gemma-4-26b-a4b-it")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")


def test_without_a_configured_provider_scaleway_answers(nothing_configured):
    llm = get_llm()

    assert llm.model_name == "gemma-4-26b-a4b-it"
    assert _base_url(llm) == "https://api.scaleway.ai/v1"


def test_without_a_configured_provider_scaleway_embeds(nothing_configured):
    from graph.llm_factory import get_embedding_model_name

    assert get_embedding_provider() == "scaleway"
    # whichever model answers
    assert get_embedding_provider("gemini") == "scaleway"
    assert get_embedding_model_name() == "bge-multilingual-gemma2"


def test_gemini_stays_available_when_chosen(nothing_configured, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

    assert get_embedding_provider() == "gemini"
    assert type(get_llm()).__name__ == "ChatGoogleGenerativeAI"
