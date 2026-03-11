"""Truncate context for grader calls to reduce token usage."""

from langchain_core.documents import Document

# Cap per-doc and total context size for any LLM grader (not for final generation).
MAX_CHARS_PER_DOC = 420
MAX_CONTEXT_CHARS = 3600

# More generous limits for the generation grader so it sees the same facts the generator used.
# Otherwise the grader may flag valid content (e.g. "geologiske prøver") as ungrounded.
MAX_CHARS_PER_DOC_GENERATION_GRADER = 1200
MAX_CONTEXT_CHARS_GENERATION_GRADER = 8000


def truncate_docs_for_grader(
    documents: list[Document],
    *,
    max_chars_per_doc: int = MAX_CHARS_PER_DOC,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Build a single context string from documents, truncating each and total to stay under token limits."""
    if not documents:
        return ""
    parts: list[str] = []
    total = 0
    for d in documents:
        text = (d.page_content or "")[:max_chars_per_doc]
        if not text:
            continue
        if total + len(text) + 2 > max_context_chars:
            remaining = max_context_chars - total - 20
            if remaining > 0:
                parts.append(text[:remaining] + "...")
            break
        parts.append(text)
        total += len(text) + 2
    return "\n\n".join(parts)
