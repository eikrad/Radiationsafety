"""LangGraph RAG workflow with optional Brave Search fallback and trusted-source verification."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from graph.chains.generation_grader import get_generation_grader
from graph.chains.truncate import truncate_docs_for_grader
from graph.i18n import detect_language, get_warning_web_search_poor
from graph.consts import (
    FINALIZE,
    GENERATE,
    GRADE_DOCUMENTS,
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


def decide_to_generate(state: GraphState) -> str:
    """Route to RETRIEVE_MISSING (then maybe WEB_SEARCH) if docs insufficient and fallback enabled, else GENERATE."""
    if state["web_search"] and env_bool("WEB_SEARCH_ENABLED"):
        return RETRIEVE_MISSING
    return GENERATE


def decide_after_retrieve_missing(state: GraphState) -> str:
    """After second retrieval: GENERATE if sufficient, else WEB_SEARCH."""
    if state.get("sufficient_after_missing"):
        return GENERATE
    return WEB_SEARCH


def grade_generation_grounded(state: GraphState) -> str:
    """Check grounding and answer quality in one call; return outcome for routing."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_search_attempted = state.get("web_search_attempted", False)
    llm = state.get("llm") or get_llm()
    grader = get_generation_grader(llm)
    docs_str = truncate_docs_for_grader(documents) if documents else "No documents"
    score = grader.invoke(
        {"documents": docs_str, "question": question, "generation": generation}
    )
    if not score.grounded:
        if env_bool("WEB_SEARCH_ENABLED") and not web_search_attempted:
            return "web_search"
        return "end"
    if score.answers_question:
        return "useful"
    if env_bool("WEB_SEARCH_ENABLED") and not web_search_attempted:
        return "web_search"
    return "end"


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
    {GENERATE: GENERATE, WEB_SEARCH: WEB_SEARCH},
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded,
    {"useful": VERIFY_TRUSTED, "web_search": WEB_SEARCH, "end": VERIFY_TRUSTED},
)
workflow.add_edge(VERIFY_TRUSTED, FINALIZE)
workflow.add_edge(FINALIZE, END)

app = workflow.compile()
