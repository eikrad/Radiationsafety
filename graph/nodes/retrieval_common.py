"""Merging retrieved chunks without duplicates (graph nodes and eval)."""

from langchain_core.documents import Document


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
