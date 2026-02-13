"""LangGraph RAG workflow with optional Brave Search fallback."""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import get_answer_grader
from graph.chains.hallucinations_grader import get_hallucination_grader
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

FINALIZE = "finalize"


def _web_search_enabled() -> bool:
    return os.getenv("WEB_SEARCH_ENABLED", "false").lower() in ("true", "1")


def decide_to_generate(state: GraphState) -> str:
    """Route to WEB_SEARCH if docs irrelevant and fallback enabled, else GENERATE."""
    if state["web_search"] and _web_search_enabled():
        return WEB_SEARCH
    return GENERATE


def grade_generation_grounded(state: GraphState) -> str:
    """Check grounding and answer quality; return outcome for routing."""
    from graph.llm_factory import get_llm

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_search_attempted = state.get("web_search_attempted", False)
    llm = state.get("llm") or get_llm()
    hall_grader = get_hallucination_grader(llm)
    ans_grader = get_answer_grader(llm)

    docs_str = (
        "\n\n".join(d.page_content for d in documents) if documents else "No documents"
    )
    hall_score = hall_grader.invoke(
        {"documents": docs_str, "generation": generation}
    )
    if not hall_score.binary_score:
        if _web_search_enabled() and not web_search_attempted:
            return "web_search"
        return "end"
    ans_score = ans_grader.invoke({"question": question, "generation": generation})
    if ans_score.binary_score:
        return "useful"
    if _web_search_enabled() and not web_search_attempted:
        return "web_search"
    return "end"


def finalize(state: GraphState) -> Dict[str, Any]:
    """Set retrieval_warning when web search was attempted but didn't yield good results."""
    warning = None
    if state.get("web_search_attempted"):
        warning = (
            "Die Websuche konnte keine ausreichend guten Quellen liefern. "
            "Die Antwort basiert möglicherweise auf unzureichenden Informationen."
        )
    return {"retrieval_warning": warning}


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)
workflow.add_node(FINALIZE, finalize)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded,
    {"useful": END, "web_search": WEB_SEARCH, "end": FINALIZE},
)
workflow.add_edge(FINALIZE, END)

app = workflow.compile()
