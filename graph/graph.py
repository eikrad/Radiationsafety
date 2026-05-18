"""LangGraph RAG workflow with optional Brave Search fallback and trusted-source verification."""

from typing import Any

from langgraph.graph import END, StateGraph

from graph.consts import (
    FINALIZE,
    GENERATE,
    GRADE_DOCUMENTS,
    GRADE_GENERATION,
    PREPARE_RETRY_RETRIEVE,
    RETRIEVE,
    RETRIEVE_MISSING,
    VERIFY_TRUSTED,
    WEB_SEARCH,
    env_bool,
)
from graph.i18n import detect_language, get_warning_web_search_poor
from graph.nodes import (
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    retrieve_missing,
    verify_trusted,
    web_search,
)
from graph.state import GenerationRoute, GraphState, RouteAfterMissing


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


def route_after_grade_generation(state: GraphState) -> GenerationRoute:
    """Pure routing — reads flags written by GRADE_GENERATION node. No LLM call."""
    if state.get("generation_passed_grading"):
        return "useful"
    return _generation_retry_route(
        web_search_enabled=bool(env_bool("WEB_SEARCH_ENABLED")),
        web_search_attempted=state.get("web_search_attempted", False),
        retry_count=state.get("retry_after_generation_count") or 0,
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
    if warning is None and web_attempted and not trusted_verified:
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
workflow.add_node(GRADE_GENERATION, grade_generation)
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
workflow.add_edge(GENERATE, GRADE_GENERATION)
workflow.add_conditional_edges(
    GRADE_GENERATION,
    route_after_grade_generation,
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
