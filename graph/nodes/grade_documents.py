"""Check if retrieved context is sufficient to answer; set web_search flag if not (one call over all chunks, no per-doc grading)."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.truncate import MAX_GRADER_CONTEXT_CHARS, format_context
from graph.llm_factory import get_llm
from graph.state import GraphState
from graph.utils import throttle_llm_if_needed


def grade_documents(
    state: GraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Run one sufficiency check on the whole retrieved chunks; set web_search=True if insufficient or no docs. Keeps all docs for generation."""
    question = state["question"]
    documents = state["documents"]
    privacy_mode = state.get("privacy_mode", False)
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    if not documents:
        return {
            "documents": [],
            "trusted_documents": [],
            "web_search": False if privacy_mode else True,
        }

    context = format_context(documents, max_context_chars=MAX_GRADER_CONTEXT_CHARS)
    throttle_llm_if_needed()
    sufficiency = get_context_sufficiency_grader(llm)
    try:
        sufficient = sufficiency.invoke(
            {"question": question, "context": context},
            config=cfg,
        ).binary_score
    except ValueError:  # no verdict even when asked again: not shown sufficient
        sufficient = False
    web_search = not sufficient

    # Privacy mode: never enable web search
    if privacy_mode:
        web_search = False

    return {
        "documents": list(documents),
        "trusted_documents": list(documents),
        "web_search": web_search,
    }
