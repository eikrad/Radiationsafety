"""Graph state for RAG pipeline."""

from typing import Any, List, NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """State for the radiation safety RAG graph."""

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    web_search_attempted: bool  # Prevent infinite web search loop
    chat_history: List[tuple[str, str]]  # (question, answer) pairs for follow-ups
    retrieval_warning: NotRequired[str]  # Set when web search didn't yield good results
    llm: NotRequired[Any]  # Chat model for generation/grading; uses env fallback if absent
    # Documents from vector DB only (IAEA + Danish); used to verify answer against trusted sources
    trusted_documents: NotRequired[List[Document]]
    # Set by verify_trusted when answer was confirmed against trusted sources; finalize then skips generic web-search warning
    trusted_verified: NotRequired[bool]
    # Set by retrieve_missing: True if after second retrieval the context was sufficient, so we skip web search
    sufficient_after_missing: NotRequired[bool]
