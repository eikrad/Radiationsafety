"""What a run retrieved from: the chunks in the configured search index."""

from pathlib import Path

import pytest

from graph import search_index


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(search_index, "CHROMA_DIR", tmp_path)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    return tmp_path


def _build(chroma_dir: Path, chunks: dict[str, list[tuple[str, str]]]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    for name, docs in chunks.items():
        collection = client.get_or_create_collection(name)
        collection.add(
            ids=[i for i, _ in docs],
            documents=[t for _, t in docs],
            embeddings=[[0.1, 0.2, 0.3]] * len(docs),
        )


IAEA, DK = search_index.collection_names("gemini")


def test_the_chunk_texts_of_both_collections_are_loaded(chroma_dir):
    _build(chroma_dir, {IAEA: [("a", "ALARA")], DK: [("b", "bilag 2"), ("c", "§ 3")]})

    texts = search_index.load_chunk_texts("gemini")

    assert texts == {IAEA: ["ALARA"], DK: ["bilag 2", "§ 3"]}


def test_the_same_chunks_give_the_same_fingerprint_whatever_their_ids(
    chroma_dir, tmp_path_factory, monkeypatch
):
    _build(chroma_dir, {IAEA: [("a", "ALARA")], DK: [("b", "bilag 2"), ("c", "§ 3")]})
    first = search_index.index_fingerprint("gemini")

    rebuilt = tmp_path_factory.mktemp("rebuilt")
    monkeypatch.setattr(search_index, "CHROMA_DIR", rebuilt)
    _build(rebuilt, {IAEA: [("x", "ALARA")], DK: [("z", "§ 3"), ("y", "bilag 2")]})

    assert search_index.index_fingerprint("gemini") == first
    assert first[DK]["chunks"] == 2


def test_rechunking_changes_the_fingerprint(chroma_dir, tmp_path_factory, monkeypatch):
    _build(chroma_dir, {IAEA: [("a", "ALARA")], DK: [("b", "bilag 2 § 3")]})
    before = search_index.index_fingerprint("gemini")

    rebuilt = tmp_path_factory.mktemp("rechunked")
    monkeypatch.setattr(search_index, "CHROMA_DIR", rebuilt)
    _build(rebuilt, {IAEA: [("a", "ALARA")], DK: [("b", "bilag 2"), ("c", "§ 3")]})
    after = search_index.index_fingerprint("gemini")

    assert after[IAEA] == before[IAEA]
    assert after[DK]["content_hash"] != before[DK]["content_hash"]


def test_a_missing_collection_is_recorded_as_missing(chroma_dir):
    _build(chroma_dir, {IAEA: [("a", "ALARA")]})

    assert search_index.index_fingerprint("gemini")[DK] is None
