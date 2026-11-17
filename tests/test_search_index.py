"""The search index: both collections of one embedding model, searched together."""

import pytest
from langchain_core.documents import Document

from graph import search_index
from graph.search_index import ChromaIndex, collection_names
from tests.fakes import RecordingEmbeddings


def _build(persist_dir, embedding_provider: str, embeddings, chunks_per_collection=6):
    from langchain_chroma import Chroma

    for name in collection_names(embedding_provider):
        Chroma.from_documents(
            [
                Document(page_content=f"{name} chunk {i}", metadata={"source": name})
                for i in range(chunks_per_collection)
            ],
            embeddings,
            collection_name=name,
            persist_directory=str(persist_dir),
        )


@pytest.fixture
def built(tmp_path):
    """A scaleway index in tmp_path, and its embeddings."""
    embeddings = RecordingEmbeddings()
    _build(tmp_path, "scaleway", embeddings)
    embeddings.tracing_seen.clear()
    return tmp_path, embeddings


def test_a_search_returns_each_collections_best_chunks(built, monkeypatch):
    persist_dir, embeddings = built
    monkeypatch.setenv("RETRIEVER_K", "3")
    index = ChromaIndex("scaleway", persist_dir=persist_dir, embeddings=embeddings)

    iaea, dk = index.search("dose limit")

    iaea_name, dk_name = collection_names("scaleway")
    assert index.k == 3
    assert len(iaea) == 3 and {d.metadata["source"] for d in iaea} == {iaea_name}
    assert len(dk) == 3 and {d.metadata["source"] for d in dk} == {dk_name}


def test_eval_can_search_deeper_than_the_graph(built):
    persist_dir, embeddings = built
    index = ChromaIndex("scaleway", persist_dir=persist_dir, embeddings=embeddings, k=2)

    iaea, dk = index.search("dose limit", k=5)

    assert (len(iaea), len(dk)) == (5, 5)


def test_a_question_kept_out_of_tracing_stays_out_in_the_search(built, monkeypatch):
    """Both collections are searched in worker threads; tracing switched off for
    the question (privacy mode, user keys) must hold there too."""
    import langsmith as ls
    from langsmith.utils import get_env_var

    persist_dir, embeddings = built
    index = ChromaIndex("scaleway", persist_dir=persist_dir, embeddings=embeddings)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    get_env_var.cache_clear()
    try:
        with ls.tracing_context(enabled=False):
            index.search("dose limit")
    finally:
        monkeypatch.delenv("LANGCHAIN_TRACING_V2")
        get_env_var.cache_clear()

    assert embeddings.tracing_seen == [False, False]


def test_a_built_index_is_ready(built):
    persist_dir, embeddings = built

    assert ChromaIndex(
        "scaleway", persist_dir=persist_dir, embeddings=embeddings
    ).ready()


def test_an_index_that_was_never_built_is_not_ready(tmp_path):
    readiness = ChromaIndex("ollama", persist_dir=tmp_path, embeddings=object()).ready()

    assert not readiness
    assert readiness.problem == "not_built"


def test_scaleway_embeddings_without_a_model_are_misconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("SCW_EMBED_MODEL", raising=False)

    readiness = ChromaIndex(
        "scaleway", persist_dir=tmp_path, embeddings=object()
    ).ready()

    assert readiness.problem == "misconfigured"
    assert "SCW_EMBED_MODEL" in readiness.detail


def test_an_index_built_by_another_model_names_the_fix(built):
    """The configured model's vectors don't fit collections another model built."""
    persist_dir, _ = built
    index = ChromaIndex(
        "scaleway", persist_dir=persist_dir, embeddings=RecordingEmbeddings(dim=7)
    )

    with pytest.raises(RuntimeError, match="ingestion.py"):
        index.search("dose limit")


def test_each_scaleway_model_searches_its_own_collections(monkeypatch):
    monkeypatch.setattr(search_index, "get_embeddings", lambda ep: object())
    search_index.refresh()
    qwen = search_index.index_for("scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")
    other = search_index.index_for("scaleway")

    assert qwen.collections != other.collections
    assert other.collections[0].endswith("qwen3-embedding-8b")
    search_index.refresh()


def test_after_a_refresh_the_index_is_opened_again(monkeypatch):
    """Ingestion replaces collections; an index opened before must not be reused."""
    monkeypatch.setattr(search_index, "get_embeddings", lambda ep: object())
    first = search_index.index_for("scaleway")

    assert search_index.index_for("scaleway") is first
    search_index.refresh()
    assert search_index.index_for("scaleway") is not first


@pytest.mark.parametrize("lang", ["en", "de", "da"])
def test_the_not_built_message_says_how_to_build_the_index(lang, monkeypatch):
    from graph.i18n import index_not_ready_message
    from graph.search_index import Readiness

    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")

    scaleway = index_not_ready_message(Readiness.not_built(), "scaleway", lang)
    local = index_not_ready_message(Readiness.not_built(), "ollama", lang)

    assert "EMBEDDING_PROVIDER=scaleway SCW_EMBED_MODEL=qwen3-embedding-8b" in scaleway
    assert "--reembed-from gemini" in scaleway
    assert "GOOGLE_API_KEY" not in scaleway
    assert "LLM_PROVIDER=ollama uv run python ingestion.py" in local


def test_a_misconfigured_index_names_the_setting(monkeypatch):
    from graph.i18n import index_not_ready_message
    from graph.search_index import Readiness

    readiness = Readiness.misconfigured("SCW_EMBED_MODEL is not set")

    assert index_not_ready_message(readiness, "scaleway", "de") == (
        "SCW_EMBED_MODEL is not set"
    )
