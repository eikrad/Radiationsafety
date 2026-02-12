"""Grade retrieved documents for relevance; set web_search flag if none relevant."""

from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Filter irrelevant docs; set web_search=True if any doc is not relevant."""
    question = state["question"]
    documents = state["documents"]

    filtered = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score:
            filtered.append(d)
        else:
            web_search = True

    return {
        "documents": filtered,
        "question": question,
        "web_search": web_search or len(filtered) == 0,
    }
