"""LangGraph RAG workflow with optional Brave Search fallback."""

import os

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucinations_grader import hallucination_grader
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def _web_search_enabled() -> bool:
    return os.getenv("WEB_SEARCH_ENABLED", "false").lower() in ("true", "1")


def decide_to_generate(state: GraphState) -> str:
    """Route to WEB_SEARCH if docs irrelevant and fallback enabled, else GENERATE."""
    if state["web_search"] and _web_search_enabled():
        return WEB_SEARCH
    return GENERATE


def grade_generation_grounded(state: GraphState) -> str:
    """Check grounding and answer quality; return outcome for routing."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    web_search_attempted = state.get("web_search_attempted", False)

    docs_str = (
        "\n\n".join(d.page_content for d in documents) if documents else "No documents"
    )
    hall_score = hallucination_grader.invoke(
        {"documents": docs_str, "generation": generation}
    )
    if not hall_score.binary_score:
        if _web_search_enabled() and not web_search_attempted:
            return "web_search"
        return "end"
    ans_score = answer_grader.invoke({"question": question, "generation": generation})
    if ans_score.binary_score:
        return "useful"
    if _web_search_enabled() and not web_search_attempted:
        return "web_search"
    return "end"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

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
    {"useful": END, "web_search": WEB_SEARCH, "end": END},
)

app = workflow.compile()
