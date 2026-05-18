"""Second retrieval targeting 'missing' information; re-check sufficiency before web search."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.missing_query_chain import invoke_missing_query_chain
from graph.chains.truncate import truncate_docs_for_grader
from graph.llm_factory import get_embedding_provider, get_llm
from graph.nodes.retrieval_common import invoke_dual_retrievers, merge_unique_documents
from graph.state import GraphState
from graph.utils import chat_context_prefix, throttle_llm_if_needed


def retrieve_missing(
    state: GraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Retrieve again with an LLM-generated query for missing info; merge with existing docs; set sufficient_after_missing."""
    question = state["question"]
    existing = list(state.get("documents") or [])
    trusted = list(state.get("trusted_documents") or [])
    chat_history = state.get("chat_history") or []
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    doc_context = "\n\n".join(d.page_content for d in existing) if existing else "None."
    context_str = (
        chat_context_prefix(chat_history) + "Document context:\n" + doc_context
    )
    reflection = state.get("reflection") or ""
    throttle_llm_if_needed()
    missing_query = invoke_missing_query_chain(
        question, context_str, llm, config=cfg, reflection=reflection
    )

    ep = state.get("embedding_provider") or get_embedding_provider()
    iaea_docs, dk_docs = invoke_dual_retrievers(
        embedding_provider=ep,
        query=missing_query,
        config=cfg,
    )
    merged, new_docs = merge_unique_documents(existing, iaea_docs + dk_docs)
    trusted_merged = list(trusted) + new_docs

    sufficient = False
    if merged:
        truncated = truncate_docs_for_grader(merged)
        throttle_llm_if_needed()
        sufficiency = get_context_sufficiency_grader(llm)
        result = sufficiency.invoke(
            {"question": question, "context": truncated},
            config=cfg,
        )
        sufficient = bool(result.binary_score)

    out: dict[str, Any] = {
        "documents": merged,
        "trusted_documents": trusted_merged,
        "sufficient_after_missing": sufficient,
    }
    # Only increment retrieval_count when in the initial flow (not retry-after-generation).
    if (state.get("retry_after_generation_count") or 0) == 0:
        out["retrieval_count"] = (state.get("retrieval_count") or 1) + 1
    return out
