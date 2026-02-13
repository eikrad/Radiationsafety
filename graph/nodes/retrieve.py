"""Retrieve documents from both IAEA and Danish law collections."""

from typing import Any, Dict

from langchain_core.documents import Document

from graph.state import GraphState
from ingestion import get_retrievers


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Query both collections, merge and deduplicate by content, return unified list."""
    question = state["question"]
    iaea_retriever, dk_retriever = get_retrievers()

    iaea_docs = iaea_retriever.invoke(question)
    dk_docs = dk_retriever.invoke(question)

    seen = set()
    merged: list[Document] = []
    for d in iaea_docs + dk_docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            merged.append(d)

    chat_history = state.get("chat_history") or []
    return {
        "documents": merged,
        "question": question,
        "web_search_attempted": False,
        "chat_history": chat_history,
    }
