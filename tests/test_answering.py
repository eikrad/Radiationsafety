"""The Answerer end to end: the real graph, a scripted model, a fake index and web."""

import httpx
import openai
import pytest
from langchain_core.documents import Document

from graph.answering import (
    Answerer,
    LocalModelUnavailable,
    ModelChoice,
    RateLimited,
    TimedOut,
)
from tests.fakes import FakeIndex, RecordingWebSearch, ScriptedChatModel

_IAEA = [
    Document(
        page_content="III.1. The occupational dose limit is 20 mSv per year.",
        metadata={"source": "GSR-3.pdf", "document_type": "IAEA"},
    ),
    Document(
        page_content="III.2. Averaged over five years, 100 mSv.",
        metadata={"source": "GSR-3.pdf", "document_type": "IAEA"},
    ),
]
_DK = [
    Document(
        page_content="§ 14. Dosisgrænsen for arbejdstagere er 20 mSv pr. år.",
        metadata={"source": "BEK-2025-138405", "document_type": "Danish law"},
    ),
]


class _Setup:
    """An Answerer over fakes, and what it touched."""

    def __init__(self, model: ScriptedChatModel, web_search_enabled: bool):
        self.model = model
        self.web = RecordingWebSearch()
        self.indexes: dict[str, FakeIndex] = {}
        self.answerer = Answerer(
            web_search_enabled=web_search_enabled,
            web_search=self.web,
            index_for=self._index_for,
            build_llm=lambda **kwargs: model,
        )

    def _index_for(self, embedding_provider: str) -> FakeIndex:
        return self.indexes.setdefault(
            embedding_provider,
            FakeIndex((_IAEA, _DK), embedding_provider=embedding_provider),
        )

    def answer(self, question, provider, **kwargs):
        return self.answerer.answer(
            question,
            ModelChoice(provider=provider, **kwargs.pop("choice", {})),
            **kwargs,
        )

    @property
    def searched(self) -> FakeIndex:
        [index] = self.indexes.values()
        return index


def _setup(model=None, web_search_enabled=True) -> _Setup:
    return _Setup(model or ScriptedChatModel(), web_search_enabled)


def _unanswerable() -> ScriptedChatModel:
    """The context never suffices and every answer fails its check, though
    what it states is grounded in the sources."""
    return ScriptedChatModel(sufficient=False, answer_passes=False)


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


def test_nothing_about_a_question_in_privacy_mode_is_traced(tracing_on):
    setup = _setup()

    answer = setup.answer("What is the occupational dose limit?", "ollama")

    assert answer.privacy_mode is True
    assert setup.model.tracing_seen and not any(setup.model.tracing_seen)
    assert setup.searched.tracing_seen and not any(setup.searched.tracing_seen)


def test_privacy_mode_searches_the_local_index():
    setup = _setup()

    setup.answer("What is the occupational dose limit?", "ollama")

    assert list(setup.indexes) == ["ollama"]


@pytest.mark.parametrize(
    ("api_keys", "traced"), [({"scaleway": "user-key"}, False), ({}, True)]
)
def test_a_question_asked_with_the_users_own_key_is_not_traced(
    tracing_on, api_keys, traced
):
    setup = _setup()

    setup.answer(
        "What is the occupational dose limit?",
        "scaleway",
        choice={"api_keys": api_keys},
    )

    assert set(setup.model.tracing_seen) == {traced}
    assert set(setup.searched.tracing_seen) == {traced}


def test_privacy_mode_never_searches_the_web():
    setup = _setup(_unanswerable(), web_search_enabled=True)

    answer = setup.answer("Which authority is told about a lost source?", "ollama")

    assert setup.web.queries == []
    assert answer.used_web_search is False


def test_the_web_is_the_last_resort_when_the_operator_allows_it():
    setup = _setup(_unanswerable(), web_search_enabled=True)

    answer = setup.answer("Which authority is told about a lost source?", "scaleway")

    assert len(setup.web.queries) >= 1
    assert answer.used_web_search is True


def test_with_web_search_off_an_incomplete_answer_says_so():
    from graph.i18n import WEB_SEARCH_OFF_NOTE

    setup = _setup(_unanswerable(), web_search_enabled=False)

    answer = setup.answer("Which authority is told about a lost source?", "scaleway")

    assert setup.web.queries == []
    assert WEB_SEARCH_OFF_NOTE["en"] in answer.warning


def test_in_privacy_mode_an_incomplete_answer_does_not_mention_web_search():
    from graph.i18n import WEB_SEARCH_OFF_NOTE

    setup = _setup(_unanswerable(), web_search_enabled=False)

    answer = setup.answer("Which authority is told about a lost source?", "ollama")

    assert answer.warning
    assert WEB_SEARCH_OFF_NOTE["en"] not in answer.warning


def test_the_trace_shows_how_the_answer_came_about():
    setup = _setup(_unanswerable(), web_search_enabled=False)

    answer = setup.answer("Which authority is told about a lost source?", "scaleway")

    trace = answer.trace
    assert {d.metadata["source"] for d in trace.initial_documents} == {
        "GSR-3.pdf",
        "BEK-2025-138405",
    }
    assert trace.sufficient is False
    assert trace.node_path[:2] == ["retrieve", "grade_documents"]
    assert "retrieve_missing" in trace.node_path
    assert trace.missing_queries and all(trace.missing_queries)
    assert setup.searched.queries[1:] == trace.missing_queries
    assert trace.context


def test_a_follow_up_question_extends_the_conversation():
    earlier = [("What is the dose limit?", "20 mSv per year.")]
    setup = _setup(ScriptedChatModel(reply="Also 20 mSv."))

    answer = setup.answer("And in Denmark?", "scaleway", history=earlier)

    assert answer.chat_history == earlier + [("And in Denmark?", "Also 20 mSv.")]


def test_sources_name_each_document_once():
    answer = _setup().answer("What is the occupational dose limit?", "scaleway")

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
def test_a_provider_failure_is_named(failure, expected):
    setup = _setup(ScriptedChatModel(error=failure))

    with pytest.raises(expected):
        setup.answer("Dose limit?", "scaleway")


def test_a_local_model_failure_says_how_to_fix_it():
    setup = _setup(ScriptedChatModel(error=ConnectionError("connection refused")))

    with pytest.raises(LocalModelUnavailable, match="ollama serve"):
        setup.answer("Dose limit?", "ollama")


def test_a_cloud_model_without_a_key_is_refused_before_anything_runs(monkeypatch):
    from graph.llm_factory import APIKeyError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(APIKeyError):
        Answerer(web_search_enabled=False).answer(
            "Dose limit?", ModelChoice(provider="openai")
        )
