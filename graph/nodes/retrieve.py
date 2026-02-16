"""Retrieve documents from both IAEA and Danish law collections."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from langchain_core.documents import Document

from graph.state import GraphState
from ingestion import get_retrievers


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Query both collections in parallel, merge and deduplicate by content."""
    question = state["question"]
    iaea_retriever, dk_retriever = get_retrievers()

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_iaea = executor.submit(iaea_retriever.invoke, question)
        fut_dk = executor.submit(dk_retriever.invoke, question)
        iaea_docs = fut_iaea.result()
        dk_docs = fut_dk.result()

    seen = set()
    merged: list[Document] = []
    for d in iaea_docs + dk_docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            merged.append(d)

    return {
        "documents": merged,
        "web_search_attempted": False,
    }
