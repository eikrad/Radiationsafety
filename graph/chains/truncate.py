"""The context the generator and every grader read: whole chunks, each with its source.

Graders used to see the first 420 (grade_documents) or 1200 (verify_trusted)
characters of each ~2500-character chunk, so facts late in a chunk were
invisible to them: grade_documents caught 0 of 3 insufficient retrievals
(#129) and local answers were flagged "not verified" (#136). Truncating the
context also changes sufficiency verdicts by itself (Sufficient Context,
Joren et al. 2025). A cap applies only to very long contexts and drops whole
chunks, never the end of one.
"""

from typing import Any

from langchain_core.documents import Document

from graph.consts import CONTEXT_SEPARATOR

# Whole chunks up to about 15k tokens; reached only after retrieve_missing has
# merged several retrievals.
MAX_GRADER_CONTEXT_CHARS = 60_000


def format_document(doc: Any) -> str:
    """One document with its source, so the model can use and cite it and tell
    Danish law from IAEA text."""
    meta = getattr(doc, "metadata", {}) or {}
    label = meta.get("source", "retrieved")
    if meta.get("document_type"):
        label = f"{label} ({meta['document_type']})"
    return f"[Source: {label}]\n{doc.page_content}"


def format_context(
    documents: list[Document], max_context_chars: int | None = None
) -> str:
    """Web results first (they are short and easily overlooked after long
    chunks), then the rest in retrieval order; whole chunks only."""
    ordered = sorted(
        documents,
        key=lambda d: (getattr(d, "metadata", {}) or {}).get("document_type") != "web",
    )
    parts: list[str] = []
    total = 0
    for doc in ordered:
        if not (doc.page_content or "").strip():
            continue
        part = format_document(doc)
        added = len(part) + (len(CONTEXT_SEPARATOR) if parts else 0)
        if max_context_chars is not None and total + added > max_context_chars:
            break
        parts.append(part)
        total += added
    return CONTEXT_SEPARATOR.join(parts)
