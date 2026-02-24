"""Truncate context for grader calls to reduce token usage."""

from langchain_core.documents import Document

# Cap per-doc and total context size for any LLM grader (not for final generation).
MAX_CHARS_PER_DOC = 420
MAX_CONTEXT_CHARS = 3600


def truncate_docs_for_grader(documents: list[Document]) -> str:
    """Build a single context string from documents, truncating each and total to stay under token limits."""
    if not documents:
        return ""
    parts: list[str] = []
    total = 0
    for d in documents:
        text = (d.page_content or "")[:MAX_CHARS_PER_DOC]
        if not text:
            continue
        if total + len(text) + 2 > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total - 20
            if remaining > 0:
                parts.append(text[:remaining] + "...")
            break
        parts.append(text)
        total += len(text) + 2
    return "\n\n".join(parts)
