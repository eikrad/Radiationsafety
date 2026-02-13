"""Graph state for RAG pipeline."""

from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """State for the radiation safety RAG graph."""

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    web_search_attempted: bool  # Prevent infinite web search loop
    chat_history: List[tuple[str, str]]  # (question, answer) pairs for follow-ups
