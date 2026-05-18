"""Grade the generation for grounding and completeness; write reflection hint to state."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.chains.generation_grader import get_generation_grader
from graph.chains.truncate import (
    MAX_CHARS_PER_DOC_GENERATION_GRADER,
    MAX_CONTEXT_CHARS_GENERATION_GRADER,
    truncate_docs_for_grader,
)
from graph.llm_factory import get_llm
from graph.state import GraphState
from graph.utils import throttle_llm_if_needed

_SENTINEL = "retry"


def grade_generation(
    state: GraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Invoke generation grader; write generation_passed_grading and reflection to state.

    reflection is set to:
      - ""         when the generation passed (passed=True)
      - missing_info when the LLM provided a hint (passed=False, missing_info non-empty)
      - "retry"    sentinel when passed=False but the LLM omitted the hint
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    context_used = state.get("context_used_for_generation")
    if context_used and context_used.strip():
        docs_str = context_used
    elif documents:
        docs_str = truncate_docs_for_grader(
            documents,
            max_chars_per_doc=MAX_CHARS_PER_DOC_GENERATION_GRADER,
            max_context_chars=MAX_CONTEXT_CHARS_GENERATION_GRADER,
        )
    else:
        docs_str = "No documents"

    throttle_llm_if_needed()
    grader = get_generation_grader(llm)
    score = grader.invoke(
        {"documents": docs_str, "question": question, "generation": generation},
        config=cfg,
    )

    if score.passed:
        return {"generation_passed_grading": True, "reflection": ""}

    reflection = score.missing_info.strip() or _SENTINEL
    return {"generation_passed_grading": False, "reflection": reflection}
