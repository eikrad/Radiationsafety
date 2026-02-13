"""Grade retrieved documents for relevance; set web_search flag if none relevant."""

from typing import Any, Dict

from graph.chains.retrieval_grader import get_retrieval_grader
from graph.llm_factory import get_llm
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Filter irrelevant docs; set web_search=True if any doc is not relevant."""
    question = state["question"]
    documents = state["documents"]
    llm = state.get("llm") or get_llm()
    grader = get_retrieval_grader(llm)

    filtered = []
    web_search = False
    for d in documents:
        score = grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score:
            filtered.append(d)
        else:
            web_search = True

    chat_history = state.get("chat_history") or []
    return {
        "documents": filtered,
        "question": question,
        "web_search": web_search or len(filtered) == 0,
        "chat_history": chat_history,
    }
