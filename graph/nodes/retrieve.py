"""Retrieve documents from both IAEA and Danish law collections."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.i18n import detect_language, get_warning_embeddings_not_built
from graph.llm_factory import get_embedding_provider
from graph.nodes.retrieval_common import invoke_dual_retrievers, merge_unique_documents
from graph.state import GraphState
from ingestion import check_embedding_collections_ready


def _retrieval_query(question: str, chat_history: list[tuple[str, str]]) -> str:
    """For follow-ups, include last exchange so retrieval has context (e.g. 'What about section 5?')."""
    if not chat_history:
        return question
    last_q, last_a = chat_history[-1]
    # Keep query length reasonable for embedding; last A truncated
    a_snippet = (last_a[:300] + "…") if len(last_a) > 300 else last_a
    return f"Previous: {last_q}. Assistant: {a_snippet}. Current question: {question}"


def retrieve(state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Query both collections in parallel, merge and deduplicate by content."""
    question = state["question"]
    chat_history = state.get("chat_history") or []
    query = _retrieval_query(question, chat_history)
    ep = state.get("embedding_provider") or get_embedding_provider()
    ready, _ = check_embedding_collections_ready(ep)
    if not ready:
        lang = detect_language(question)
        return {
            "documents": [],
            "trusted_documents": [],
            "web_search_attempted": False,
            "retrieval_warning": get_warning_embeddings_not_built(ep, lang),
            "retrieval_count": 1,
        }
    def _map_retrieval_error(e: Exception) -> Exception:
        try:
            if "dimension" in str(e).lower() and "embedding" in str(e).lower():
                return RuntimeError(
                    "Embedding dimension mismatch: the Chroma collection was built with a different "
                    "embedding model. Re-run full ingestion (set GOOGLE_API_KEY in .env, then "
                    "uv run python ingestion.py) so the vector store uses Gemini embeddings."
                )
            return e
        except Exception:
            return e

    iaea_docs, dk_docs = invoke_dual_retrievers(
        embedding_provider=ep,
        query=query,
        config=config,
        map_error=_map_retrieval_error,
    )
    merged, _ = merge_unique_documents([], iaea_docs + dk_docs)

    return {
        "documents": merged,
        "trusted_documents": list(merged),
        "web_search_attempted": False,
        "retrieval_count": 1,
    }
