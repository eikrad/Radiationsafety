"""Ingestion pipeline tests."""

import ingestion
from ingestion import (
    DK_LAW_COLLECTION,
    IAEA_COLLECTION,
    get_collection_names,
    load_dk_law_docs,
    load_iaea_docs,
)


def test_load_iaea_docs_returns_empty_when_no_dirs(tmp_path, monkeypatch):
    """load_iaea_docs returns [] when IAEA dirs do not exist."""
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_iaea_docs()
    assert docs == []


def test_load_iaea_docs_returns_empty_for_empty_iaea_dir(tmp_path, monkeypatch):
    """load_iaea_docs returns [] when IAEA dir exists but has no PDFs."""
    iaea = tmp_path / "IAEA"
    iaea.mkdir()
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_iaea_docs()
    assert docs == []


def test_load_dk_law_docs_returns_empty_when_no_dir(tmp_path, monkeypatch):
    """load_dk_law_docs returns [] when Bekendtgørelse does not exist."""
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_dk_law_docs()
    assert docs == []


def test_load_dk_law_docs_returns_empty_for_empty_dir(tmp_path, monkeypatch):
    """load_dk_law_docs returns [] when Bekendtgørelse exists but has no PDFs."""
    dk = tmp_path / "Bekendtgørelse"
    dk.mkdir()
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_dk_law_docs()
    assert docs == []


def test_rotate_backups_keeps_only_keep_newest(tmp_path):
    """rotate_backups deletes older files so only `keep` most recent remain."""
    import time

    from ingestion import rotate_backups

    (tmp_path / "a_1.xml").write_text("1")
    time.sleep(0.02)
    (tmp_path / "a_2.xml").write_text("2")
    time.sleep(0.02)
    (tmp_path / "a_3.xml").write_text("3")
    rotate_backups(tmp_path, "a", keep=2)
    remaining = sorted(tmp_path.glob("a_*.xml"), key=lambda p: p.stat().st_mtime)
    assert len(remaining) == 2
    assert (tmp_path / "a_1.xml").exists() is False


def test_collection_names():
    """Collection names match expected constants."""
    assert IAEA_COLLECTION == "radiation-iaea"
    assert DK_LAW_COLLECTION == "radiation-dk-law"


def test_get_collection_names_by_provider():
    """get_collection_names returns base names for gemini, -mistral suffix for mistral."""
    iaea_g, dk_g = get_collection_names("gemini")
    assert iaea_g == "radiation-iaea"
    assert dk_g == "radiation-dk-law"
    iaea_m, dk_m = get_collection_names("mistral")
    assert iaea_m == "radiation-iaea-mistral"
    assert dk_m == "radiation-dk-law-mistral"


# --- One collection pair per embedding model ---------------------------------

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


class _FakeEmbeddings(Embeddings):
    """Deterministic vectors of a fixed size, so collections built by different
    'models' are told apart by their dimension."""

    def __init__(self, dim: int, fail_first: int = 0):
        self.dim, self.fail_first, self.calls = dim, fail_first, 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("transient")
        return [[float(len(t))] + [1.0] * (self.dim - 1) for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "_CHROMA_DIR", tmp_path)
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")
    ingestion.clear_retrievers_cache()
    yield tmp_path
    ingestion.clear_retrievers_cache()


def _build_gemini_collections(chroma_dir):
    from langchain_chroma import Chroma

    iaea, dk = get_collection_names("gemini")
    docs = {
        iaea: [
            Document(
                page_content="III.1. For occupational exposure …",
                metadata={"source": "GSR-3.pdf", "document_type": "IAEA"},
            ),
            Document(
                page_content="523. The TI for a package …",
                metadata={"source": "SSR-6.pdf", "document_type": "IAEA"},
            ),
        ],
        dk: [
            Document(
                page_content="§ 14. For erhvervsmæssig bestråling …",
                metadata={"source": "B20250138405.pdf", "document_type": "Danish law"},
            )
        ],
    }
    for name, batch in docs.items():
        Chroma.from_documents(
            batch,
            _FakeEmbeddings(3),
            collection_name=name,
            persist_directory=str(chroma_dir),
        )
    return docs


def _collection(chroma_dir, name):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir)).get_collection(name)


def test_each_scaleway_model_gets_its_own_collections(monkeypatch):
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")
    qwen = get_collection_names("scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    bge = get_collection_names("scaleway")

    assert qwen == (
        "radiation-iaea-scw-qwen3-embedding-8b",
        "radiation-dk-law-scw-qwen3-embedding-8b",
    )
    assert bge != qwen
    assert get_collection_names("gemini") == ("radiation-iaea", "radiation-dk-law")


def test_reembedding_copies_the_same_chunks_with_new_vectors(chroma_dir, monkeypatch):
    source = _build_gemini_collections(chroma_dir)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: _FakeEmbeddings(4))

    ingestion.reembed_from("gemini", target="scaleway")

    for src, dst in zip(
        get_collection_names("gemini"), get_collection_names("scaleway"), strict=True
    ):
        copied = _collection(chroma_dir, dst).get(
            include=["documents", "metadatas", "embeddings"]
        )
        assert sorted(copied["documents"]) == sorted(
            d.page_content for d in source[src]
        )
        assert sorted(m["source"] for m in copied["metadatas"]) == sorted(
            d.metadata["source"] for d in source[src]
        )
        assert {len(e) for e in copied["embeddings"]} == {4}
    # the source stays untouched
    assert _collection(chroma_dir, get_collection_names("gemini")[0]).count() == 2


def test_reembedding_replaces_an_earlier_copy(chroma_dir, monkeypatch):
    _build_gemini_collections(chroma_dir)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: _FakeEmbeddings(4))

    ingestion.reembed_from("gemini", target="scaleway")
    ingestion.reembed_from("gemini", target="scaleway")

    assert _collection(chroma_dir, get_collection_names("scaleway")[0]).count() == 2


def test_a_transient_embedding_failure_is_retried(chroma_dir, monkeypatch):
    _build_gemini_collections(chroma_dir)
    flaky = _FakeEmbeddings(4, fail_first=1)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: flaky)
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)

    ingestion.reembed_from("gemini", target="scaleway")

    assert _collection(chroma_dir, get_collection_names("scaleway")[0]).count() == 2


def test_missing_scaleway_collections_say_how_to_build_them(chroma_dir):
    ready, message = ingestion.check_embedding_collections_ready("scaleway")

    assert ready is False
    assert "EMBEDDING_PROVIDER=scaleway" in message
    assert "--reembed-from gemini" in message


def test_retrievers_for_different_scaleway_models_are_not_mixed_up(
    chroma_dir, monkeypatch
):
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: _FakeEmbeddings(4))

    qwen_iaea, _ = ingestion.get_retrievers("scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    bge_iaea, _ = ingestion.get_retrievers("scaleway")

    assert qwen_iaea.vectorstore._collection.name.endswith("qwen3-embedding-8b")
    assert bge_iaea.vectorstore._collection.name.endswith("bge-multilingual-gemma2")


@pytest.mark.parametrize("lang", ["en", "de", "da"])
def test_the_user_facing_warning_for_missing_scaleway_embeddings_names_scaleway(lang):
    from graph.i18n import get_warning_embeddings_not_built

    msg = get_warning_embeddings_not_built("scaleway", lang)

    assert "Scaleway" in msg
    assert "GOOGLE_API_KEY" not in msg
