"""LangGraph RAG workflow with optional Brave Search fallback and trusted-source verification."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from graph.chains.generation_grader import get_generation_grader
from graph.chains.truncate import (
    MAX_CHARS_PER_DOC_GENERATION_GRADER,
    MAX_CONTEXT_CHARS_GENERATION_GRADER,
    truncate_docs_for_grader,
)
from graph.i18n import detect_language, get_warning_web_search_poor
from graph.consts import (
    FINALIZE,
    GENERATE,
    GRADE_DOCUMENTS,
    PREPARE_RETRY_RETRIEVE,
    RETRIEVE,
    RETRIEVE_MISSING,
    VERIFY_TRUSTED,
    WEB_SEARCH,
    env_bool,
)
from graph.llm_factory import get_llm
from graph.nodes import (
    generate,
    grade_documents,
    retrieve,
    retrieve_missing,
    verify_trusted,
    web_search,
)
from graph.state import GraphState
from graph.utils import throttle_llm_if_needed


def decide_to_generate(state: GraphState) -> str:
    """Route to RETRIEVE_MISSING (then maybe WEB_SEARCH) if docs insufficient and fallback enabled, else GENERATE."""
    if state["web_search"] and env_bool("WEB_SEARCH_ENABLED"):
        return RETRIEVE_MISSING
    return GENERATE


def decide_after_retrieve_missing(state: GraphState) -> str:
    """After retrieval: GENERATE if sufficient or in retry-after-generation path; else RETRIEVE_MISSING (up to 3 total) or WEB_SEARCH."""
    if (state.get("retry_after_generation_count") or 0) > 0:
        return GENERATE
    if state.get("sufficient_after_missing"):
        return GENERATE
    if (state.get("retrieval_count") or 1) < 3:
        return RETRIEVE_MISSING
    return WEB_SEARCH


def grade_generation_grounded(state: GraphState) -> str:
    """Check grounding and answer quality in one call; return outcome for routing. Up to 2 retry retrievals before web search."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_search_attempted = state.get("web_search_attempted", False)
    retry_count = state.get("retry_after_generation_count") or 0
    llm = state.get("llm") or get_llm()
    throttle_llm_if_needed()
    grader = get_generation_grader(llm)
    # Use the exact context the generator saw, so the grader can verify all claims (e.g. geologiske prøver).
    context_used = state.get("context_used_for_generation")
    if context_used is not None and context_used.strip():
        docs_str = context_used
    elif documents:
        docs_str = truncate_docs_for_grader(
            documents,
            max_chars_per_doc=MAX_CHARS_PER_DOC_GENERATION_GRADER,
            max_context_chars=MAX_CONTEXT_CHARS_GENERATION_GRADER,
        )
    else:
        docs_str = "No documents"
    score = grader.invoke(
        {"documents": docs_str, "question": question, "generation": generation}
    )
    def _would_use_web_search() -> bool:
        return bool(env_bool("WEB_SEARCH_ENABLED") and not web_search_attempted)
    if not score.grounded:
        if _would_use_web_search():
            return "retry_retrieve" if retry_count < 2 else "web_search"
        return "end"
    if score.answers_question:
        return "useful"
    if _would_use_web_search():
        return "retry_retrieve" if retry_count < 2 else "web_search"
    return "end"


def prepare_retry_retrieve(state: GraphState) -> Dict[str, Any]:
    """Increment retry_after_generation_count before another RETRIEVE_MISSING (max 2 before web search)."""
    count = (state.get("retry_after_generation_count") or 0) + 1
    return {"retry_after_generation_count": count}


def finalize(state: GraphState) -> Dict[str, Any]:
    """Set retrieval_warning if not already set by verify_trusted. Message in the language of the question."""
    warning = state.get("retrieval_warning")
    if warning is None and state.get("web_search_attempted") and not state.get("trusted_verified"):
        lang = detect_language(state.get("question") or "")
        warning = get_warning_web_search_poor(lang)
    return {"retrieval_warning": warning}


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(RETRIEVE_MISSING, retrieve_missing)
workflow.add_node(PREPARE_RETRY_RETRIEVE, prepare_retry_retrieve)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)
workflow.add_node(VERIFY_TRUSTED, verify_trusted)
workflow.add_node(FINALIZE, finalize)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {RETRIEVE_MISSING: RETRIEVE_MISSING, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    RETRIEVE_MISSING,
    decide_after_retrieve_missing,
    {GENERATE: GENERATE, RETRIEVE_MISSING: RETRIEVE_MISSING, WEB_SEARCH: WEB_SEARCH},
)
workflow.add_edge(PREPARE_RETRY_RETRIEVE, RETRIEVE_MISSING)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded,
    {
        "useful": VERIFY_TRUSTED,
        "retry_retrieve": PREPARE_RETRY_RETRIEVE,
        "web_search": WEB_SEARCH,
        "end": VERIFY_TRUSTED,
    },
)
workflow.add_edge(VERIFY_TRUSTED, FINALIZE)
workflow.add_edge(FINALIZE, END)

app = workflow.compile()
