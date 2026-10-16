"""invoke_dual_retrievers: the one retrieval path shared by the graph and eval."""

from langchain_core.documents import Document

import graph.nodes.retrieval_common as retrieval_common


class _FakeRetriever:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def invoke(self, query, config=None, **kwargs):
        self.calls.append(kwargs)
        return [Document(page_content=f"{self.name}: {query}")]


def _patch(monkeypatch):
    iaea, dk = _FakeRetriever("iaea"), _FakeRetriever("dk")
    monkeypatch.setattr(retrieval_common, "get_retrievers", lambda ep: (iaea, dk))
    return iaea, dk


def test_the_graph_retrieves_with_the_retrievers_own_k(monkeypatch):
    iaea, dk = _patch(monkeypatch)

    found = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None
    )

    assert [d.page_content for d in found[0]] == ["iaea: q"]
    assert iaea.calls == [{}] and dk.calls == [{}]


def test_eval_can_retrieve_deeper_through_the_same_path(monkeypatch):
    iaea, dk = _patch(monkeypatch)

    retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None, k=20
    )

    assert iaea.calls == [{"k": 20}] and dk.calls == [{"k": 20}]


class _RankedRetriever:
    """Returns `k` chunks per call, numbered in embedding order."""

    def __init__(self, name, default_k=5):
        self.name = name
        self.default_k = default_k
        self.calls = []

    def invoke(self, query, config=None, **kwargs):
        self.calls.append(kwargs)
        k = kwargs.get("k", self.default_k)
        return [Document(page_content=f"{self.name} {i}") for i in range(1, k + 1)]


class _ReverseScorer:
    """Scores the last embedding candidate highest."""

    def __init__(self):
        self.queries = []

    def scores(self, query, texts):
        self.queries.append(query)
        return [float(text.split()[-1]) for text in texts]


def _patch_ranked(monkeypatch, scorer, candidates=20):
    iaea, dk = _RankedRetriever("iaea"), _RankedRetriever("dk")
    monkeypatch.setattr(retrieval_common, "get_retrievers", lambda ep: (iaea, dk))
    monkeypatch.setattr(retrieval_common, "get_reranker", lambda: scorer)
    monkeypatch.setattr(retrieval_common, "rerank_candidates", lambda: candidates)
    monkeypatch.setattr(retrieval_common, "RETRIEVER_K", 5)
    return iaea, dk


def test_a_reranker_picks_the_top_k_of_each_collections_candidates(monkeypatch):
    scorer = _ReverseScorer()
    iaea, dk = _patch_ranked(monkeypatch, scorer)

    found_iaea, found_dk = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None
    )

    assert iaea.calls == [{"k": 20}] and dk.calls == [{"k": 20}]
    assert [d.page_content for d in found_iaea] == [
        "iaea 20",
        "iaea 19",
        "iaea 18",
        "iaea 17",
        "iaea 16",
    ]
    assert [d.page_content for d in found_dk][0] == "dk 20"
    assert scorer.queries == ["q", "q"]


def test_eval_deeper_than_the_candidates_reranks_its_whole_depth(monkeypatch):
    iaea, _ = _patch_ranked(monkeypatch, _ReverseScorer(), candidates=10)

    found_iaea, _ = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None, k=20
    )

    assert iaea.calls == [{"k": 20}]
    assert len(found_iaea) == 20 and found_iaea[0].page_content == "iaea 20"


def test_a_failing_reranker_keeps_the_embedding_order(monkeypatch, caplog):
    class _Broken:
        def scores(self, query, texts):
            raise RuntimeError("model not loaded")

    _patch_ranked(monkeypatch, _Broken())

    found_iaea, _ = retrieval_common.invoke_dual_retrievers(
        embedding_provider="scaleway", query="q", config=None
    )

    assert [d.page_content for d in found_iaea] == [f"iaea {i}" for i in range(1, 6)]
    assert "re-rank" in caplog.text.lower()
