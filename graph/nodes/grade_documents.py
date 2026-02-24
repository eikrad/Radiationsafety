"""Check if retrieved context is sufficient to answer; set web_search flag if not (no per-doc grading to save tokens)."""

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.truncate import truncate_docs_for_grader
from graph.llm_factory import get_llm
from graph.state import GraphState


def grade_documents(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Run one sufficiency check on truncated context; set web_search=True if insufficient or no docs. Keeps all docs for generation."""
    question = state["question"]
    documents = state["documents"]
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    if not documents:
        return {
            "documents": [],
            "trusted_documents": [],
            "web_search": True,
        }

    truncated = truncate_docs_for_grader(documents)
    sufficiency = get_context_sufficiency_grader(llm)
    sufficient = sufficiency.invoke(
        {"question": question, "context": truncated},
        config=cfg,
    )
    web_search = not sufficient.binary_score

    return {
        "documents": list(documents),
        "trusted_documents": list(documents),
        "web_search": web_search,
    }
