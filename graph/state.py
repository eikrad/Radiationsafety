"""Graph state for RAG pipeline."""

from typing import Any, List, NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """State for the radiation safety RAG graph."""

    question: str
    generation: str
    # Exact context string passed to the generator; grader uses this so it sees the same facts.
    context_used_for_generation: NotRequired[str]
    web_search: bool
    documents: List[Document]
    web_search_attempted: bool  # Prevent infinite web search loop
    chat_history: List[tuple[str, str]]  # (question, answer) pairs for follow-ups
    retrieval_warning: NotRequired[str]  # Set when web search didn't yield good results or Mistral embeddings missing
    embedding_provider: NotRequired[str]  # "gemini" | "mistral"; from request model (gemini/openai → gemini)
    llm: NotRequired[Any]  # Chat model for generation/grading; uses env fallback if absent
    # Documents from vector DB only (IAEA + Danish); used to verify answer against trusted sources
    trusted_documents: NotRequired[List[Document]]
    # Set by verify_trusted when answer was confirmed against trusted sources; finalize then skips generic web-search warning
    trusted_verified: NotRequired[bool]
    # Set by retrieve_missing: True if after second retrieval the context was sufficient, so we skip web search
    sufficient_after_missing: NotRequired[bool]
    # Number of vector-store retrievals so far (1 after RETRIEVE; 2–3 after RETRIEVE_MISSING). Used to allow up to 3 before WEB_SEARCH.
    retrieval_count: NotRequired[int]
    # Retry count after generation grader failed: 0 → 2 retrievals allowed before WEB_SEARCH.
    retry_after_generation_count: NotRequired[int]
