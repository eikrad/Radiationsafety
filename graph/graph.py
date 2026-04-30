"""LangGraph RAG workflow with optional Brave Search fallback and trusted-source verification."""

from typing import Any

from langgraph.graph import END, StateGraph

from graph.chains.generation_grader import get_generation_grader
from graph.chains.truncate import (
    MAX_CHARS_PER_DOC_GENERATION_GRADER,
    MAX_CONTEXT_CHARS_GENERATION_GRADER,
    truncate_docs_for_grader,
)
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
from graph.i18n import detect_language, get_warning_web_search_poor
from graph.llm_factory import get_llm
from graph.nodes import (
    generate,
    grade_documents,
    retrieve,
    retrieve_missing,
    verify_trusted,
    web_search,
)
from graph.state import GenerationRoute, GraphState, RouteAfterMissing
from graph.utils import throttle_llm_if_needed


def decide_to_generate(state: GraphState) -> str:
    """Route to RETRIEVE_MISSING (then maybe WEB_SEARCH) if docs insufficient and fallback enabled, else GENERATE."""
    if state["web_search"] and env_bool("WEB_SEARCH_ENABLED"):
        return RETRIEVE_MISSING
    return GENERATE


def decide_after_retrieve_missing(state: GraphState) -> RouteAfterMissing:
    """After retrieval: GENERATE if sufficient or in retry-after-generation path; else RETRIEVE_MISSING (up to 3 total) or WEB_SEARCH."""
    if (state.get("retry_after_generation_count") or 0) > 0:
        return GENERATE
    if state.get("sufficient_after_missing"):
        return GENERATE
    if (state.get("retrieval_count") or 1) < 3:
        return RETRIEVE_MISSING
    return WEB_SEARCH


def _generation_retry_route(
    *,
    web_search_enabled: bool,
    web_search_attempted: bool,
    retry_count: int,
) -> GenerationRoute:
    """Centralize generation retry routing for clearer behavior contract."""
    if not web_search_enabled or web_search_attempted:
        return "end"
    return "retry_retrieve" if retry_count < 2 else "web_search"


def grade_generation_grounded(state: GraphState) -> GenerationRoute:
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

    web_search_enabled = bool(env_bool("WEB_SEARCH_ENABLED"))

    if not score.grounded:
        return _generation_retry_route(
            web_search_enabled=web_search_enabled,
            web_search_attempted=web_search_attempted,
            retry_count=retry_count,
        )
    if score.answers_question:
        return "useful"
    return _generation_retry_route(
        web_search_enabled=web_search_enabled,
        web_search_attempted=web_search_attempted,
        retry_count=retry_count,
    )


def prepare_retry_retrieve(state: GraphState) -> dict[str, Any]:
    """Increment retry_after_generation_count before another RETRIEVE_MISSING (max 2 before web search)."""
    count = (state.get("retry_after_generation_count") or 0) + 1
    return {"retry_after_generation_count": count}


def finalize(state: GraphState) -> dict[str, Any]:
    """Set retrieval_warning if not already set by verify_trusted. Message in the language of the question."""
    warning = state.get("retrieval_warning")
    web_attempted = bool(state.get("web_search_attempted"))
    trusted_verified = bool(state.get("trusted_verified"))
    if (
        warning is None
        and web_attempted
        and not trusted_verified
    ):
        lang = detect_language(state.get("question") or "")
        warning = get_warning_web_search_poor(lang)
    if web_attempted:
        outcome = "web_search_verified" if trusted_verified else "web_search_unverified"
    else:
        outcome = (
            "trusted_only_verified" if trusted_verified else "trusted_only_unverified"
        )
    return {"retrieval_warning": warning, "routing_outcome": outcome}


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
