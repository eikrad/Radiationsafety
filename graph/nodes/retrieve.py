"""Retrieve documents from both IAEA and Danish law collections."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.state import GraphState
from ingestion import get_retrievers


def _retrieval_query(question: str, chat_history: list[tuple[str, str]]) -> str:
    """For follow-ups, include last exchange so retrieval has context (e.g. 'What about section 5?')."""
    if not chat_history:
        return question
    last_q, last_a = chat_history[-1]
    # Keep query length reasonable for embedding; last A truncated
    a_snippet = (last_a[:300] + "…") if len(last_a) > 300 else last_a
    return f"Previous: {last_q}. Assistant: {a_snippet}. Current question: {question}"


def retrieve(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Query both collections in parallel, merge and deduplicate by content."""
    question = state["question"]
    chat_history = state.get("chat_history") or []
    query = _retrieval_query(question, chat_history)
    cfg = config or {}
    iaea_retriever, dk_retriever = get_retrievers()

    def _invoke_iaea():
        return iaea_retriever.invoke(query, config=cfg)

    def _invoke_dk():
        return dk_retriever.invoke(query, config=cfg)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_iaea = executor.submit(_invoke_iaea)
        fut_dk = executor.submit(_invoke_dk)
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
        "trusted_documents": list(merged),
        "web_search_attempted": False,
    }
