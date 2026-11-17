"""The Answerer end to end: the real graph, a scripted model, a temp-dir search index."""

import httpx
import openai
import pytest

from graph.answering import (
    Answerer,
    LocalModelUnavailable,
    ModelChoice,
    RateLimited,
    TimedOut,
)
from tests.fakes import ScriptedChatModel


@pytest.fixture
def tracing_on(monkeypatch):
    """LangSmith tracing switched on for the server, pointed at nowhere."""
    from langsmith.utils import get_env_var

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://127.0.0.1:9")
    get_env_var.cache_clear()
    yield
    for name in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    get_env_var.cache_clear()


def _answerer(model: ScriptedChatModel) -> Answerer:
    return Answerer(build_llm=lambda **kwargs: model)


def test_nothing_about_a_question_in_privacy_mode_is_traced(
    temp_search_index, tracing_on
):
    model = ScriptedChatModel()

    answer = _answerer(model).answer(
        "What is the occupational dose limit?", ModelChoice(provider="ollama")
    )

    assert answer.privacy_mode is True
    assert model.tracing_seen and not any(model.tracing_seen)
    assert temp_search_index.tracing_seen and not any(temp_search_index.tracing_seen)


@pytest.mark.parametrize(
    ("api_keys", "traced"), [({"scaleway": "user-key"}, False), ({}, True)]
)
def test_a_question_asked_with_the_users_own_key_is_not_traced(
    temp_search_index, tracing_on, api_keys, traced
):
    model = ScriptedChatModel()

    _answerer(model).answer(
        "What is the occupational dose limit?",
        ModelChoice(provider="scaleway", api_keys=api_keys),
    )

    assert set(model.tracing_seen) == {traced}
    assert set(temp_search_index.tracing_seen) == {traced}


def _unanswerable() -> ScriptedChatModel:
    """The context never suffices and every answer fails its check, though
    what it states is grounded in the sources."""
    return ScriptedChatModel(sufficient=False, answer_passes=False)


def test_privacy_mode_never_searches_the_web(
    temp_search_index, web_queries, monkeypatch
):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?", ModelChoice(provider="ollama")
    )

    assert web_queries == []
    assert answer.used_web_search is False


def test_the_web_is_the_last_resort_when_the_operator_allows_it(
    temp_search_index, web_queries, monkeypatch
):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?",
        ModelChoice(provider="scaleway"),
    )

    assert len(web_queries) >= 1
    assert answer.used_web_search is True


def test_with_web_search_off_an_incomplete_answer_says_so(
    temp_search_index, web_queries, monkeypatch
):
    from graph.i18n import WEB_SEARCH_OFF_NOTE

    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?",
        ModelChoice(provider="scaleway"),
    )

    assert web_queries == []
    assert WEB_SEARCH_OFF_NOTE["en"] in answer.warning


def test_in_privacy_mode_an_incomplete_answer_does_not_mention_web_search(
    temp_search_index, web_queries, monkeypatch
):
    from graph.i18n import WEB_SEARCH_OFF_NOTE

    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?", ModelChoice(provider="ollama")
    )

    assert answer.warning
    assert WEB_SEARCH_OFF_NOTE["en"] not in answer.warning


def test_the_trace_shows_how_the_answer_came_about(temp_search_index, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?",
        ModelChoice(provider="scaleway"),
    )

    trace = answer.trace
    assert {d.metadata["source"] for d in trace.initial_documents} == {
        "GSR-3.pdf",
        "BEK-2025-138405",
    }
    assert trace.sufficient is False
    assert trace.node_path[:2] == ["retrieve", "grade_documents"]
    assert "retrieve_missing" in trace.node_path
    assert trace.missing_queries and all(trace.missing_queries)
    assert trace.context


def test_a_follow_up_question_extends_the_conversation(temp_search_index):
    earlier = [("What is the dose limit?", "20 mSv per year.")]

    answer = _answerer(ScriptedChatModel(reply="Also 20 mSv.")).answer(
        "And in Denmark?", ModelChoice(provider="scaleway"), history=earlier
    )

    assert answer.chat_history == earlier + [("And in Denmark?", "Also 20 mSv.")]


def test_sources_name_each_document_once(temp_search_index):
    answer = _answerer(ScriptedChatModel()).answer(
        "What is the occupational dose limit?", ModelChoice(provider="scaleway")
    )

    assert len(answer.documents) == 3
    assert answer.sources == [
        ("GSR-3.pdf", "IAEA"),
        ("BEK-2025-138405", "Danish law"),
    ]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"), RateLimited),
        (TimeoutError("read timed out"), TimedOut),
        (
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://api.scaleway.ai/v1/chat")
            ),
            TimedOut,
        ),
    ],
)
def test_a_provider_failure_is_named(temp_search_index, failure, expected):
    model = ScriptedChatModel(error=failure)

    with pytest.raises(expected):
        _answerer(model).answer("Dose limit?", ModelChoice(provider="scaleway"))


def test_a_local_model_failure_says_how_to_fix_it(temp_search_index):
    model = ScriptedChatModel(error=ConnectionError("connection refused"))

    with pytest.raises(LocalModelUnavailable, match="ollama serve"):
        _answerer(model).answer("Dose limit?", ModelChoice(provider="ollama"))


def test_a_cloud_model_without_a_key_is_refused_before_anything_runs(monkeypatch):
    from graph.llm_factory import APIKeyError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(APIKeyError):
        Answerer().answer("Dose limit?", ModelChoice(provider="openai"))
