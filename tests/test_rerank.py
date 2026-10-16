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


class _FakeCrossEncoder:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.pairs.append(list(pairs))
        return [float(len(passage)) for _, passage in pairs]


@pytest.fixture
def cross_encoder(monkeypatch):
    """The bge cross-encoder with its model replaced; records each load."""
    loads = []
    model = _FakeCrossEncoder()

    def _load(name, device):
        loads.append((name, device))
        return model

    monkeypatch.setenv("RERANKER", "bge-reranker-v2-m3")
    monkeypatch.setattr(rerank, "_load_cross_encoder", _load)
    monkeypatch.setattr(rerank, "_cuda_available", lambda: True)
    rerank._loaded.clear()
    yield model, loads
    rerank._loaded.clear()


def test_the_cross_encoder_scores_each_question_chunk_pair(cross_encoder):
    model, loads = cross_encoder

    scores = rerank.get_reranker("scaleway").scores("q", ["ab", "abcd"])

    assert scores == [2.0, 4.0]
    assert model.pairs == [[("q", "ab"), ("q", "abcd")]]
    assert loads == [("BAAI/bge-reranker-v2-m3", "cuda")]


def test_the_cross_encoder_is_loaded_once(cross_encoder):
    _, loads = cross_encoder

    rerank.get_reranker("scaleway").scores("q", ["a"])
    rerank.get_reranker("scaleway").scores("q", ["b"])

    assert len(loads) == 1


def test_privacy_mode_reranks_on_the_cpu(cross_encoder):
    """The local answer model keeps the GPU, as with bge-m3's question embeddings."""
    _, loads = cross_encoder

    rerank.get_reranker("ollama").scores("q", ["a"])

    assert loads == [("BAAI/bge-reranker-v2-m3", "cpu")]
