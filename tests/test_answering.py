"""The Answerer end to end: the real graph, a scripted model, a temp-dir search index."""

import importlib

import pytest
from langchain_core.documents import Document

import ingestion
from graph.answering import (
    Answerer,
    LocalModelUnavailable,
    ModelChoice,
    RateLimited,
    TimedOut,
)
from tests.fakes import RecordingEmbeddings, ScriptedChatModel

_CHUNKS = {
    "iaea": [
        Document(
            page_content="III.1. The occupational dose limit is 20 mSv per year.",
            metadata={"source": "GSR-3.pdf", "document_type": "IAEA"},
        ),
        Document(
            page_content="III.2. Averaged over five years, 100 mSv.",
            metadata={"source": "GSR-3.pdf", "document_type": "IAEA"},
        ),
    ],
    "dk": [
        Document(
            page_content="§ 14. Dosisgrænsen for arbejdstagere er 20 mSv pr. år.",
            metadata={"source": "BEK-2025-138405", "document_type": "Danish law"},
        ),
    ],
}


@pytest.fixture
def embeddings(tmp_path, monkeypatch):
    """A search index in tmp_path for the cloud and the local embeddings."""
    from langchain_chroma import Chroma

    recording = RecordingEmbeddings()
    monkeypatch.setattr(ingestion, "_CHROMA_DIR", tmp_path)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: recording)
    ingestion.clear_retrievers_cache()
    for provider in ("scaleway", "ollama"):
        iaea, dk = ingestion.get_collection_names(provider)
        for name, chunks in ((iaea, _CHUNKS["iaea"]), (dk, _CHUNKS["dk"])):
            Chroma.from_documents(
                chunks, recording, collection_name=name, persist_directory=str(tmp_path)
            )
    recording.tracing_seen.clear()
    yield recording
    ingestion.clear_retrievers_cache()


@pytest.fixture
def web(monkeypatch):
    """Every query web search sends; the web itself is never reached."""
    queries: list[str] = []

    class RecordingBrave:
        @classmethod
        def from_api_key(cls, api_key, search_kwargs=None):
            return cls()

        def invoke(self, query):
            queries.append(query)
            return "[]"

    # the module, not the node function graph.nodes re-exports under its name
    web_search_module = importlib.import_module("graph.nodes.web_search")
    monkeypatch.setattr(web_search_module, "BraveSearch", RecordingBrave)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    return queries


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


def test_nothing_about_a_question_in_privacy_mode_is_traced(embeddings, tracing_on):
    model = ScriptedChatModel()

    answer = _answerer(model).answer(
        "What is the occupational dose limit?", ModelChoice(provider="ollama")
    )

    assert answer.privacy_mode is True
    assert model.tracing_seen and not any(model.tracing_seen)
    assert embeddings.tracing_seen and not any(embeddings.tracing_seen)


@pytest.mark.parametrize(
    ("api_keys", "traced"), [({"scaleway": "user-key"}, False), ({}, True)]
)
def test_a_question_asked_with_the_users_own_key_is_not_traced(
    embeddings, tracing_on, api_keys, traced
):
    model = ScriptedChatModel()

    _answerer(model).answer(
        "What is the occupational dose limit?",
        ModelChoice(provider="scaleway", api_keys=api_keys),
    )

    assert set(model.tracing_seen) == {traced}
    assert set(embeddings.tracing_seen) == {traced}


def _unanswerable() -> ScriptedChatModel:
    """The context never suffices and every answer fails its check, though
    what it states is grounded in the sources."""
    return ScriptedChatModel(sufficient=False, answer_passes=False)


def test_privacy_mode_never_searches_the_web(embeddings, web, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?", ModelChoice(provider="ollama")
    )

    assert web == []
    assert answer.used_web_search is False


def test_the_web_is_the_last_resort_when_the_operator_allows_it(
    embeddings, web, monkeypatch
):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?",
        ModelChoice(provider="scaleway"),
    )

    assert len(web) >= 1
    assert answer.used_web_search is True


def test_with_web_search_off_an_incomplete_answer_says_so(embeddings, web, monkeypatch):
    from graph.i18n import WEB_SEARCH_OFF_NOTE

    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?",
        ModelChoice(provider="scaleway"),
    )

    assert web == []
    assert WEB_SEARCH_OFF_NOTE["en"] in answer.warning


def test_in_privacy_mode_an_incomplete_answer_does_not_mention_web_search(
    embeddings, web, monkeypatch
):
    from graph.i18n import WEB_SEARCH_OFF_NOTE

    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    answer = _answerer(_unanswerable()).answer(
        "Which authority is told about a lost source?", ModelChoice(provider="ollama")
    )

    assert answer.warning
    assert WEB_SEARCH_OFF_NOTE["en"] not in answer.warning


def test_the_trace_shows_how_the_answer_came_about(embeddings, monkeypatch):
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


def test_a_follow_up_question_extends_the_conversation(embeddings):
    earlier = [("What is the dose limit?", "20 mSv per year.")]

    answer = _answerer(ScriptedChatModel(reply="Also 20 mSv.")).answer(
        "And in Denmark?", ModelChoice(provider="scaleway"), history=earlier
    )

    assert answer.chat_history == earlier + [("And in Denmark?", "Also 20 mSv.")]


def test_sources_name_each_document_once(embeddings):
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
    ],
)
def test_a_provider_failure_is_named(embeddings, failure, expected):
    model = ScriptedChatModel(error=failure)

    with pytest.raises(expected):
        _answerer(model).answer("Dose limit?", ModelChoice(provider="scaleway"))


def test_a_local_model_failure_says_how_to_fix_it(embeddings):
    model = ScriptedChatModel(error=ConnectionError("connection refused"))

    with pytest.raises(LocalModelUnavailable, match="ollama serve"):
        _answerer(model).answer("Dose limit?", ModelChoice(provider="ollama"))


def test_a_cloud_model_without_a_key_is_refused_before_anything_runs(monkeypatch):
    from graph.llm_factory import APIKeyError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(APIKeyError):
        Answerer().answer("Dose limit?", ModelChoice(provider="openai"))
