"""Optional re-ranking of retrieval candidates (#146).

Each collection returns RERANK_CANDIDATES chunks by embedding similarity; a
re-ranker scores every (question, chunk) pair and the best RETRIEVER_K go on.
Off unless RERANKER names a re-ranker; the adoption rule is in ROADMAP.md,
"Re-ranking retrieval candidates".
"""

import os
from collections.abc import Callable
from typing import Protocol

_OFF = {"", "off", "none"}


class Reranker(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text; higher is more relevant."""
        ...


def reranker_name() -> str | None:
    """The configured re-ranker (RERANKER), or None when re-ranking is off."""
    raw = (os.getenv("RERANKER") or "").strip()
    return None if raw.lower() in _OFF else raw


def rerank_candidates() -> int:
    """Chunks per collection the re-ranker chooses from (RERANK_CANDIDATES, default 20)."""
    raw = (os.getenv("RERANK_CANDIDATES") or "").strip()
    n = int(raw) if raw else 20
    if n < 1:
        raise ValueError("RERANK_CANDIDATES must be at least 1")
    return n


# Chunks are cut at about 1500 characters; 1024 tokens holds question and chunk.
_CROSS_ENCODER_MAX_LENGTH = 1024


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _load_cross_encoder(model: str, device: str):
    """Half precision on the GPU: about 3x faster and half the memory (1.1 GB
    for bge-reranker-v2-m3), with the same ranking."""
    import torch
    from sentence_transformers import CrossEncoder

    model_kwargs = {"dtype": torch.float16} if device == "cuda" else None
    return CrossEncoder(
        model,
        max_length=_CROSS_ENCODER_MAX_LENGTH,
        device=device,
        model_kwargs=model_kwargs,
    )


class _CrossEncoderReranker:
    def __init__(self, model):
        self._model = model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


def _cross_encoder(model: str) -> Callable[[str], Reranker]:
    return lambda device: _CrossEncoderReranker(_load_cross_encoder(model, device))


def _http_post(url: str, headers: dict, payload: dict) -> dict:
    import httpx

    from graph.llm_factory import request_timeout

    response = httpx.post(url, headers=headers, json=payload, timeout=request_timeout())
    response.raise_for_status()
    return response.json()


class _ScalewayReranker:
    """Scaleway's /v1/rerank. With an embedding model its score is the cosine of
    question and chunk embeddings (checked against /v1/embeddings), and no
    instruction is added server-side, so the question carries the model's own."""

    def __init__(self, model: str):
        self._model = model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        from graph.llm_factory import (
            _QUERY_TASK,
            _SCALEWAY_BASE_URL,
            APIKeyError,
            query_instruction_template,
        )

        key = os.getenv("SCW_SECRET_KEY")
        if not key:
            raise APIKeyError("Scaleway")
        base_url = (os.getenv("SCW_BASE_URL") or "").strip() or _SCALEWAY_BASE_URL
        template = query_instruction_template(self._model)
        reply = _http_post(
            f"{base_url.rstrip('/')}/rerank",
            {"Authorization": f"Bearer {key}"},
            {
                "model": self._model,
                "query": (
                    template.format(task=_QUERY_TASK, query=query)
                    if template
                    else query
                ),
                "documents": texts,
                "top_n": len(texts),
            },
        )
        scores = [0.0] * len(texts)
        for result in reply["results"]:
            scores[result["index"]] = float(result["relevance_score"])
        return scores


# RERANKER value -> factory taking the device.
_BACKENDS: dict[str, Callable[[str], Reranker]] = {
    "bge-reranker-v2-m3": _cross_encoder("BAAI/bge-reranker-v2-m3"),
    "scaleway:qwen3-embedding-8b": lambda device: _ScalewayReranker(
        "qwen3-embedding-8b"
    ),
}
# Re-rankers that send question and chunks off the machine: never in privacy mode.
_CLOUD = {"scaleway:qwen3-embedding-8b"}
# (RERANKER, device) -> loaded re-ranker; a model is loaded once per process.
_loaded: dict[tuple[str, str], Reranker] = {}


def _device(embedding_provider: str | None) -> str:
    """The GPU if there is one, except in privacy mode (Ollama embeddings): there
    the local answer model needs the card, as with bge-m3's question embeddings."""
    if embedding_provider == "ollama" or not _cuda_available():
        return "cpu"
    return "cuda"


def get_reranker(embedding_provider: str | None = None) -> Reranker | None:
    """The configured re-ranker, or None when re-ranking is off (and for a cloud
    re-ranker in privacy mode, which keeps every request on the machine)."""
    name = reranker_name()
    if name is None:
        return None
    if name not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"RERANKER={name!r} is not a known re-ranker ({known}, off)")
    if name in _CLOUD and embedding_provider == "ollama":
        return None
    key = (name, _device(embedding_provider))
    if key not in _loaded:
        _loaded[key] = _BACKENDS[name](key[1])
    return _loaded[key]
