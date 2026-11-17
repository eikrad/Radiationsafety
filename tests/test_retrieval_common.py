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


def test_a_question_kept_out_of_tracing_stays_out_in_the_retrievals(monkeypatch):
    """Both retrievals run in worker threads; tracing switched off for the
    question (privacy mode, user keys) must hold there too."""
    import langsmith as ls
    from langsmith.utils import get_env_var, tracing_is_enabled

    seen = []

    class _Recording(_FakeRetriever):
        def invoke(self, query, config=None, **kwargs):
            seen.append(bool(tracing_is_enabled()))
            return super().invoke(query, config, **kwargs)

    monkeypatch.setattr(
        retrieval_common,
        "get_retrievers",
        lambda ep: (_Recording("iaea"), _Recording("dk")),
    )
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    get_env_var.cache_clear()
    try:
        with ls.tracing_context(enabled=False):
            retrieval_common.invoke_dual_retrievers(
                embedding_provider="ollama", query="q", config=None
            )
    finally:
        monkeypatch.delenv("LANGCHAIN_TRACING_V2")
        get_env_var.cache_clear()

    assert seen == [False, False]
