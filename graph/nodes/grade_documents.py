"""Check if the retrieved context is sufficient to answer (one call over all chunks, no per-doc grading)."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.chains.context_sufficiency_grader import (
    context_gap,
    get_context_sufficiency_grader,
)
from graph.chains.truncate import MAX_GRADER_CONTEXT_CHARS, format_context
from graph.llm_factory import get_llm
from graph.state import GraphState
from graph.utils import throttle_llm_if_needed


def grade_documents(
    state: GraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Run one sufficiency check on the whole retrieved chunks; keeps all docs for generation."""
    question = state["question"]
    documents = state["documents"]
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    if not documents:
        return {
            "documents": [],
            "trusted_documents": [],
            "context_sufficient": False,
            "context_gap": "",
        }

    context = format_context(documents, max_context_chars=MAX_GRADER_CONTEXT_CHARS)
    throttle_llm_if_needed()
    sufficiency = get_context_sufficiency_grader(llm)
    try:
        verdict = sufficiency.invoke(
            {"question": question, "context": context},
            config=cfg,
        )
        sufficient = bool(verdict.binary_score)
        gap = "" if sufficient else context_gap(verdict)
    except ValueError:  # no verdict even when asked again: not shown sufficient
        sufficient, gap = False, ""

    return {
        "documents": list(documents),
        "trusted_documents": list(documents),
        "context_sufficient": sufficient,
        "context_gap": gap,
    }
