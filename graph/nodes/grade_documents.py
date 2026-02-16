"""Grade retrieved documents for relevance; set web_search flag if none relevant."""

from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def grade_one(doc):
        score = grader.invoke({"question": question, "document": doc.page_content})
        return doc, score.binary_score

    filtered = []
    web_search = False
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(documents)))) as executor:
        futures = {executor.submit(grade_one, d): d for d in documents}
        for fut in as_completed(futures):
            doc, relevant = fut.result()
            if relevant:
                filtered.append(doc)
            else:
                web_search = True

    return {
        "documents": filtered,
        "web_search": web_search or len(filtered) == 0,
    }
