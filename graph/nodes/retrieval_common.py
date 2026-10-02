"""Shared retrieval helpers for graph nodes."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.consts import env_bool
from graph.lexical import collection_index, reciprocal_rank_fusion
from ingestion import RETRIEVER_K, get_collection_names, get_retrievers

# Chunks each ranker contributes to the fusion; fixed, not tuned. A chunk at
# rank 50 still adds 1/110 to a score whose maximum per ranker is 1/61.
FUSION_CANDIDATES = 50

_LANGUAGES = ("english", "danish")  # IAEA, Danish law


def hybrid_enabled() -> bool:
    """Dense retrieval fused with BM25 (HYBRID_RETRIEVAL, off by default)."""
    return env_bool("HYBRID_RETRIEVAL")


def lexical_index(embedding_provider: str, which: int):
    """BM25 index of the IAEA (0) or Danish law (1) collection."""
    name = get_collection_names(embedding_provider)[which]
    return collection_index(name, _LANGUAGES[which])


def make_doc_key(doc: Document) -> str:
    """Create a stable dedupe key using source metadata + normalized content prefix."""
    meta = getattr(doc, "metadata", {}) or {}
    source = str(meta.get("source") or "")
    dtype = str(meta.get("document_type") or "")
    content = " ".join((doc.page_content or "").split())
    prefix = content[:240]
    return f"{source}|{dtype}|{prefix}"


def merge_unique_documents(
    existing_docs: list[Document], new_docs: list[Document]
) -> tuple[list[Document], list[Document]]:
    """Merge docs while preserving order; returns (merged_docs, newly_added_docs)."""
    seen = {make_doc_key(d) for d in existing_docs}
    merged = list(existing_docs)
    added: list[Document] = []
    for doc in new_docs:
        key = make_doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        added.append(doc)
    return merged, added


def invoke_dual_retrievers(
    *,
    embedding_provider: str,
    query: str,
    config: RunnableConfig | None,
    map_error: Callable[[Exception], Exception] | None = None,
    k: int | None = None,
) -> tuple[list[Document], list[Document]]:
    """Invoke IAEA and DK retrievers in parallel and return both result lists.

    k: chunks per collection; default the retrievers' own (RETRIEVER_K). Eval
    retrieves deeper through this same path, so it measures what the graph does.
    With HYBRID_RETRIEVAL each collection's dense and BM25 rankings are fused
    (reciprocal rank fusion) before the top k are taken.
    """
    iaea_retriever, dk_retriever = get_retrievers(embedding_provider)
    cfg = config or {}
    hybrid = hybrid_enabled()
    wanted = k if k is not None else RETRIEVER_K
    if hybrid:
        search = {"k": max(wanted, FUSION_CANDIDATES)}
    else:
        search = {"k": k} if k is not None else {}

    def _retrieve(retriever, which: int) -> list[Document]:
        dense = retriever.invoke(query, config=cfg, **search)
        if not hybrid:
            return dense
        lexical = lexical_index(embedding_provider, which).search(
            query, max(wanted, FUSION_CANDIDATES)
        )
        return reciprocal_rank_fusion([dense, lexical])[:wanted]

    def _invoke_safe(fn):
        try:
            return fn()
        except Exception as exc:
            if map_error is not None:
                raise map_error(exc) from exc
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_iaea = executor.submit(
            lambda: _invoke_safe(lambda: _retrieve(iaea_retriever, 0))
        )
        fut_dk = executor.submit(
            lambda: _invoke_safe(lambda: _retrieve(dk_retriever, 1))
        )
        return fut_iaea.result(), fut_dk.result()
