"""Graph state for RAG pipeline."""

from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

RouteAfterMissing = Literal["generate", "retrieve_missing", "web_search"]
GenerationRoute = Literal["useful", "retry_retrieve", "web_search", "end"]
QueryOutcome = Literal[
    "trusted_only_verified",
    "trusted_only_unverified",
    "web_search_verified",
    "web_search_unverified",
]


class GraphState(TypedDict):
    """State for the radiation safety RAG graph."""

    question: str
    generation: str
    # Exact context string passed to the generator; grader uses this so it sees the same facts.
    context_used_for_generation: NotRequired[str]
    web_search: bool
    documents: list[Document]
    web_search_attempted: bool  # Prevent infinite web search loop
    chat_history: list[tuple[str, str]]  # (question, answer) pairs for follow-ups
    retrieval_warning: NotRequired[
        str
    ]  # Set when web search didn't yield good results or Mistral embeddings missing
    embedding_provider: NotRequired[
        str
    ]  # "gemini" | "mistral"; from request model (gemini/openai → gemini)
    llm: NotRequired[
        BaseChatModel
    ]  # Chat model for generation/grading; uses env fallback if absent
    # Documents from vector DB only (IAEA + Danish); used to verify answer against trusted sources
    trusted_documents: NotRequired[list[Document]]
    # Set by verify_trusted when answer was confirmed against trusted sources; finalize then skips generic web-search warning
    trusted_verified: NotRequired[bool]
    # Set by retrieve_missing: True if after second retrieval the context was sufficient, so we skip web search
    sufficient_after_missing: NotRequired[bool]
    # Number of vector-store retrievals so far (1 after RETRIEVE; 2–3 after RETRIEVE_MISSING). Used to allow up to 3 before WEB_SEARCH.
    retrieval_count: NotRequired[int]
    # Retry count after generation grader failed: 0 → 2 retrievals allowed before WEB_SEARCH.
    retry_after_generation_count: NotRequired[int]
    routing_outcome: NotRequired[QueryOutcome]
    # Reflexion: verbal hint from the grader about what was missing in the last failed generation.
    # Passed to retrieve_missing to focus the next retrieval query.
    # Empty string when generation passed grading; "retry" sentinel when LLM omits the hint.
    # Reset to "" by the generate node on every new generation attempt.
    reflection: NotRequired[str]
    # Written by GRADE_GENERATION node; read by route_after_grade_generation.
    # True = generation passed; False = needs retry/web-search/end.
    generation_passed_grading: NotRequired[bool]
