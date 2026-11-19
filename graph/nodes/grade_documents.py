"""Check if the retrieved context is sufficient to answer (one call over all chunks, no per-doc grading)."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.llm_factory import get_llm
from graph.state import GraphState
from graph.sufficiency import check_sufficiency


def grade_documents(
    state: GraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Run one sufficiency check on the whole retrieved chunks; keeps all docs for generation."""
    question = state["question"]
    documents = state["documents"]
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    verdict = check_sufficiency(question, documents, llm, config=cfg)

    return {
        "documents": list(documents or []),
        "trusted_documents": list(documents or []),
        "context_sufficient": verdict.sufficient if verdict is not None else False,
        "context_gap": verdict.gap if verdict is not None else "",
    }
