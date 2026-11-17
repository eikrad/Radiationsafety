"""The LangGraph RAG workflow, built for one web mode.

The web mode is fixed when the graph is built (graph.answering.Answerer picks
it): "on" searches the web as the last resort, "off" never does because the
operator switched it off, "private" never does because privacy mode keeps
everything on the machine. Graphs for "off" and "private" have no web search
node and no web search to hand to any node.
"""

from functools import partial
from typing import Any, Literal

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
)
from graph.i18n import (
    detect_language,
    get_warning_answer_may_be_incomplete,
    get_warning_web_search_poor,
)
from graph.nodes import (
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    retrieve_missing,
    verify_trusted,
)
from graph.nodes import web_search as web_search_node
from graph.search_index import SearchIndex
from graph.state import GenerationRoute, GraphState, RouteAfterMissing
from graph.web import WebSearch

WebMode = Literal["on", "off", "private"]


def decide_to_generate(state: GraphState) -> str:
    """Search the database again (RETRIEVE_MISSING) if the context is insufficient, else GENERATE.

    Independent of web search: with it off the loop used to be skipped entirely."""
    if state.get("context_sufficient") is False:
        return RETRIEVE_MISSING
    return GENERATE


def decide_after_retrieve_missing(
    state: GraphState, *, web_search_allowed: bool
) -> RouteAfterMissing:
    """After retrieval: GENERATE if sufficient or in retry-after-generation path; else RETRIEVE_MISSING (up to 3 total) or WEB_SEARCH."""
    if (state.get("retry_after_generation_count") or 0) > 0:
        return GENERATE
    if state.get("sufficient_after_missing"):
        return GENERATE
    if (state.get("retrieval_count") or 1) < 3:
        return RETRIEVE_MISSING
    return WEB_SEARCH if web_search_allowed else GENERATE


def _generation_retry_route(
    *,
    web_search_allowed: bool,
    web_search_attempted: bool,
    retry_count: int,
) -> GenerationRoute:
    """Retry from the database twice, then the web if allowed, then show the answer."""
    if web_search_attempted:
        return "end"
    if retry_count < 2:
        return "retry_retrieve"
    return "web_search" if web_search_allowed else "end"


def route_after_grade_generation(
    state: GraphState, *, web_search_allowed: bool
) -> GenerationRoute:
    """Pure routing — reads flags written by GRADE_GENERATION node. No LLM call."""
    if state.get("generation_passed_grading"):
        return "useful"
    return _generation_retry_route(
        web_search_allowed=web_search_allowed,
        web_search_attempted=state.get("web_search_attempted", False),
        retry_count=state.get("retry_after_generation_count") or 0,
    )


def prepare_retry_retrieve(state: GraphState) -> dict[str, Any]:
    """Increment retry_after_generation_count before another RETRIEVE_MISSING (max 2 before web search)."""
    count = (state.get("retry_after_generation_count") or 0) + 1
    return {"retry_after_generation_count": count}


def finalize(state: GraphState, *, web_mode: WebMode) -> dict[str, Any]:
    """Set retrieval_warning if not already set by verify_trusted. Message in the language of the question.

    An answer the grader still fails after the database retries, without web
    search, gets a hint that the documents may not cover the whole question;
    with web search switched off it says so (privacy mode never has it, so
    there it is not worth mentioning)."""
    warning = state.get("retrieval_warning")
    web_attempted = bool(state.get("web_search_attempted"))
    trusted_verified = bool(state.get("trusted_verified"))
    lang = detect_language(state.get("question") or "")
    if warning is None and web_attempted and not trusted_verified:
        warning = get_warning_web_search_poor(lang)
    elif (
        warning is None
        and not web_attempted
        and state.get("generation_passed_grading") is False
    ):
        warning = get_warning_answer_may_be_incomplete(
            lang, web_search_off=web_mode == "off"
        )
    if web_attempted:
        outcome = "web_search_verified" if trusted_verified else "web_search_unverified"
    else:
        outcome = (
            "trusted_only_verified" if trusted_verified else "trusted_only_unverified"
        )
    return {"retrieval_warning": warning, "routing_outcome": outcome}


def build_graph(
    web_mode: WebMode, index: SearchIndex, web_search: WebSearch | None = None
):
    """The compiled graph for one web mode, searching `index`.

    web_search: required for "on", refused for "off" and "private", so a graph
    that must stay off the web is never handed a way onto it.
    """
    web_on = web_mode == "on"
    if web_on and web_search is None:
        raise ValueError('web mode "on" needs a web search')
    if not web_on and web_search is not None:
        raise ValueError(f'web mode "{web_mode}" takes no web search')

    workflow = StateGraph(GraphState)
    workflow.add_node(RETRIEVE, partial(retrieve, index=index))
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(RETRIEVE_MISSING, partial(retrieve_missing, index=index))
    workflow.add_node(PREPARE_RETRY_RETRIEVE, prepare_retry_retrieve)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(GRADE_GENERATION, grade_generation)
    workflow.add_node(
        VERIFY_TRUSTED, partial(verify_trusted, trusted_search=web_search)
    )
    workflow.add_node(FINALIZE, partial(finalize, web_mode=web_mode))

    workflow.set_entry_point(RETRIEVE)
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {RETRIEVE_MISSING: RETRIEVE_MISSING, GENERATE: GENERATE},
    )
    after_missing = {GENERATE: GENERATE, RETRIEVE_MISSING: RETRIEVE_MISSING}
    after_grading = {
        "useful": VERIFY_TRUSTED,
        "retry_retrieve": PREPARE_RETRY_RETRIEVE,
        "end": VERIFY_TRUSTED,
    }
    if web_on:
        workflow.add_node(WEB_SEARCH, partial(web_search_node, search=web_search))
        workflow.add_edge(WEB_SEARCH, GENERATE)
        after_missing[WEB_SEARCH] = WEB_SEARCH
        after_grading["web_search"] = WEB_SEARCH
    workflow.add_conditional_edges(
        RETRIEVE_MISSING,
        partial(decide_after_retrieve_missing, web_search_allowed=web_on),
        after_missing,
    )
    workflow.add_edge(PREPARE_RETRY_RETRIEVE, RETRIEVE_MISSING)
    workflow.add_edge(GENERATE, GRADE_GENERATION)
    workflow.add_conditional_edges(
        GRADE_GENERATION,
        partial(route_after_grade_generation, web_search_allowed=web_on),
        after_grading,
    )
    workflow.add_edge(VERIFY_TRUSTED, FINALIZE)
    workflow.add_edge(FINALIZE, END)
    return workflow.compile()
