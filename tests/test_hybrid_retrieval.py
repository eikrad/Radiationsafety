"""Dense retrieval fused with BM25 by reciprocal rank (#131), behind HYBRID_RETRIEVAL."""

from langchain_core.documents import Document

import graph.nodes.retrieval_common as retrieval_common
from graph.lexical import LexicalIndex, analyze, reciprocal_rank_fusion


def _doc(id_, text="text"):
    return Document(id=id_, page_content=text)


# --- the analyzer: a real stemmer and stop words per language -------------------


def test_danish_inflections_meet_in_one_stem():
    assert analyze("dosisgrænserne", "danish") == analyze("dosisgrænser", "danish")


def test_stop_words_go_and_numbers_and_paragraphs_stay():
    tokens = analyze("Hvad er en sikkerhedsvurdering efter § 3, nr. 54?", "danish")

    assert "er" not in tokens and "en" not in tokens
    assert {"3", "54"} <= set(tokens)
    assert analyze("the limits of exposure", "english") == ["limit", "exposur"]


# --- BM25 over one collection ------------------------------------------------------


def test_the_chunk_that_names_the_term_ranks_first():
    index = LexicalIndex(
        [
            _doc("a", "Radon: radioaktiv luftart. Referenceniveau: et niveau."),
            _doc("b", "Sikkerhedsvurdering: en vurdering af sikkerheden ved et anlæg."),
            _doc("c", "Dosisgrænser for arbejdstagere."),
        ],
        "danish",
    )

    assert [d.id for d in index.search("Hvad er en sikkerhedsvurdering?", 3)] == ["b"]


def test_chunks_sharing_no_term_with_the_question_are_not_ranked():
    index = LexicalIndex([_doc("a", "radon"), _doc("b", "transport")], "danish")

    assert index.search("dosisgrænse", 5) == []


# --- reciprocal rank fusion (Cormack et al. 2009, k = 60) -------------------------


def test_a_chunk_ranked_well_by_both_beats_one_ranked_first_by_one():
    dense = [_doc("x"), _doc("both"), _doc("y")]
    lexical = [_doc("both"), _doc("z")]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert [d.id for d in fused][:2] == ["both", "x"]
    assert len(fused) == 4  # each chunk once


def test_ties_keep_the_dense_order():
    fused = reciprocal_rank_fusion([[_doc("d1"), _doc("d2")], [_doc("l1"), _doc("l2")]])

    assert [d.id for d in fused] == ["d1", "l1", "d2", "l2"]


# --- the switch in the shared retrieval path -----------------------------------------


class _FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def invoke(self, query, config=None, **kwargs):
        self.calls.append(kwargs)
        return self.docs[: kwargs.get("k", 3)]


class _FakeIndex:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query, n):
        return self.docs[:n]


def _patch(monkeypatch, *, hybrid):
    monkeypatch.setenv("HYBRID_RETRIEVAL", "true" if hybrid else "false")
    dense = [_doc(f"d{i}") for i in range(60)]
    iaea, dk = _FakeRetriever(dense), _FakeRetriever(dense)
    monkeypatch.setattr(retrieval_common, "get_retrievers", lambda ep: (iaea, dk))
    lexical = _FakeIndex([_doc("term"), _doc("d1")])
    monkeypatch.setattr(retrieval_common, "lexical_index", lambda ep, which: lexical)
    return iaea, dk


def test_switched_off_retrieval_is_dense_only(monkeypatch):
    iaea, _ = _patch(monkeypatch, hybrid=False)

    found_iaea, _ = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None
    )

    assert [d.id for d in found_iaea] == ["d0", "d1", "d2"]
    assert iaea.calls == [{}]


def test_switched_on_a_chunk_found_only_by_bm25_can_enter_the_top_k(monkeypatch):
    iaea, dk = _patch(monkeypatch, hybrid=True)

    found_iaea, found_dk = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None
    )

    assert [d.id for d in found_iaea] == ["d1", "d0", "term"]
    assert len(found_dk) == 3
    # dense candidates come from deeper than k, so fusion has a list to work on
    assert iaea.calls == [{"k": retrieval_common.FUSION_CANDIDATES}]


def test_eval_depth_beyond_the_candidates_still_gets_that_many(monkeypatch):
    iaea, _ = _patch(monkeypatch, hybrid=True)

    found_iaea, _ = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None, k=55
    )

    assert len(found_iaea) == 55
    assert iaea.calls == [{"k": 55}]


def test_eval_runs_record_whether_retrieval_was_hybrid(monkeypatch):
    from eval.run_eval import _embedding_config

    monkeypatch.setenv("HYBRID_RETRIEVAL", "true")
    assert _embedding_config()["hybrid_retrieval"] is True
    monkeypatch.setenv("HYBRID_RETRIEVAL", "false")
    assert _embedding_config()["hybrid_retrieval"] is False


# --- against a real Chroma collection ---------------------------------------------


class _ConstantEmbeddings:
    """Every text gets the same vector, so dense order is storage order."""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def test_a_chunk_found_by_both_rankers_is_one_chunk_in_chroma(tmp_path, monkeypatch):
    import chromadb

    import ingestion
    from graph.lexical import clear_indexes

    monkeypatch.setattr(ingestion, "_CHROMA_DIR", tmp_path)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep: _ConstantEmbeddings())
    monkeypatch.setenv("HYBRID_RETRIEVAL", "true")
    ingestion.clear_retrievers_cache()
    clear_indexes()
    client = chromadb.PersistentClient(path=str(tmp_path))
    iaea_name, dk_name = ingestion.get_collection_names("gemini")
    client.get_or_create_collection(iaea_name).add(
        ids=["i1"], documents=["ALARA"], embeddings=[[0.1, 0.2, 0.3]]
    )
    texts = [f"§ {n}. Andet emne nummer {n}." for n in range(1, 9)]
    texts.append("§ 9. Sikkerhedsvurdering: en vurdering af sikkerheden.")
    client.get_or_create_collection(dk_name).add(
        ids=[f"d{n}" for n in range(1, 10)],
        documents=texts,
        embeddings=[[0.1, 0.2, 0.3]] * 9,
    )

    try:
        _, found_dk = retrieval_common.invoke_dual_retrievers(
            embedding_provider="gemini", query="Hvad er en sikkerhedsvurdering?", config=None
        )
    finally:
        ingestion.clear_retrievers_cache()

    ids = [d.id for d in found_dk]
    assert "d9" in ids and len(ids) == len(set(ids)) == 3
