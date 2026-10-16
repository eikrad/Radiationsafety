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


# RERANKER value -> factory; the backends register themselves below.
_BACKENDS: dict[str, Callable[[], Reranker]] = {}


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


def get_reranker() -> Reranker | None:
    """The configured re-ranker, or None when re-ranking is off."""
    name = reranker_name()
    if name is None:
        return None
    if name not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS)) or "none yet"
        raise ValueError(f"RERANKER={name!r} is not a known re-ranker ({known}, off)")
    return _BACKENDS[name]()
