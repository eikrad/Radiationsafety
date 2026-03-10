"""Retrieve documents from both IAEA and Danish law collections."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.llm_factory import get_embedding_provider
from graph.state import GraphState
from graph.i18n import detect_language, get_warning_mistral_embeddings_not_built
from ingestion import check_embedding_collections_ready, get_retrievers


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
    ep = state.get("embedding_provider") or get_embedding_provider()
    if ep == "mistral":
        ready, _ = check_embedding_collections_ready(ep)
        if not ready:
            lang = detect_language(question)
            return {
                "documents": [],
                "trusted_documents": [],
                "web_search_attempted": False,
                "retrieval_warning": get_warning_mistral_embeddings_not_built(lang),
            }
    iaea_retriever, dk_retriever = get_retrievers(ep)

    def _invoke_iaea():
        try:
            return iaea_retriever.invoke(query, config=cfg)
        except Exception as e:
            if "dimension" in str(e).lower() and "embedding" in str(e).lower():
                raise RuntimeError(
                    "Embedding dimension mismatch: the Chroma collection was built with a different "
                    "embedding model (e.g. Mistral 1024 dim). Re-run full ingestion with the current "
                    "LLM_PROVIDER (e.g. uv run python ingestion.py with LLM_PROVIDER=gemini in .env) "
                    "so the vector store uses the same embeddings."
                ) from e
            raise

    def _invoke_dk():
        try:
            return dk_retriever.invoke(query, config=cfg)
        except Exception as e:
            if "dimension" in str(e).lower() and "embedding" in str(e).lower():
                raise RuntimeError(
                    "Embedding dimension mismatch: the Chroma collection was built with a different "
                    "embedding model (e.g. Mistral 1024 dim). Re-run full ingestion with the current "
                    "LLM_PROVIDER (e.g. uv run python ingestion.py with LLM_PROVIDER=gemini in .env) "
                    "so the vector store uses the same embeddings."
                ) from e
            raise

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
