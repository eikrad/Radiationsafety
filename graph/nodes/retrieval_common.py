"""Shared retrieval helpers for graph nodes."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from ingestion import get_retrievers


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
    """
    iaea_retriever, dk_retriever = get_retrievers(embedding_provider)
    cfg = config or {}
    search = {"k": k} if k is not None else {}

    def _invoke_safe(fn):
        try:
            return fn()
        except Exception as exc:
            if map_error is not None:
                raise map_error(exc) from exc
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_iaea = executor.submit(
            lambda: _invoke_safe(
                lambda: iaea_retriever.invoke(query, config=cfg, **search)
            )
        )
        fut_dk = executor.submit(
            lambda: _invoke_safe(
                lambda: dk_retriever.invoke(query, config=cfg, **search)
            )
        )
        return fut_iaea.result(), fut_dk.result()
