"""Re-ranker configuration: off unless RERANKER names one."""

import pytest

import graph.rerank as rerank


@pytest.mark.parametrize("value", [None, "", "off", "OFF", "none"])
def test_reranking_is_off_unless_configured(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RERANKER", raising=False)
    else:
        monkeypatch.setenv("RERANKER", value)

    assert rerank.reranker_name() is None
    assert rerank.get_reranker() is None


def test_candidates_default_to_twenty_per_collection(monkeypatch):
    monkeypatch.delenv("RERANK_CANDIDATES", raising=False)
    assert rerank.rerank_candidates() == 20
    monkeypatch.setenv("RERANK_CANDIDATES", "30")
    assert rerank.rerank_candidates() == 30


def test_an_unknown_reranker_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("RERANKER", "no-such-model")
    with pytest.raises(ValueError, match="RERANKER"):
        rerank.get_reranker()
