"""A stalled provider call gives up instead of holding the request for minutes.

The OpenAI client alone waits up to 600 s per attempt and retries twice, so a
hanging Scaleway call kept /query open for up to half an hour.
"""

import pytest

from graph.llm_factory import get_embeddings, get_llm


def _timeout(llm) -> float:
    for name in ("request_timeout", "timeout"):
        value = getattr(llm, name, None)
        if value is not None:
            return value
    return llm.client_kwargs["timeout"]


@pytest.mark.parametrize("provider", ["scaleway", "openai", "gemini", "mistral"])
def test_cloud_models_give_up_on_a_stalled_call(provider):
    llm = get_llm(provider=provider)

    assert _timeout(llm) == 60
    assert llm.max_retries == 1


def test_the_limits_can_be_changed_in_env(monkeypatch):
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SEC", "15")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")

    llm = get_llm(provider="scaleway")

    assert _timeout(llm) == 15
    assert llm.max_retries == 0


def test_an_unreadable_limit_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SEC", "soon")

    assert _timeout(get_llm(provider="openai")) == 60


def test_local_ollama_gets_more_time_on_slow_hardware():
    assert _timeout(get_llm(provider="ollama")) == 300


def test_scaleway_search_embeddings_give_up_too(monkeypatch):
    # Without the query-instruction wrapper, get_embeddings returns the client itself.
    monkeypatch.setenv("EMBED_QUERY_INSTRUCTION", "false")
    client = get_embeddings("scaleway")

    assert client.request_timeout == 60
    assert client.max_retries == 1
