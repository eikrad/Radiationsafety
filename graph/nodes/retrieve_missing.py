"""Second retrieval targeting 'missing' information; re-check sufficiency before web search."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.chains.missing_query_chain import invoke_missing_query_chain
from graph.llm_factory import get_llm
from graph.nodes.retrieval_common import merge_unique_documents
from graph.search_index import SearchIndex
from graph.state import GraphState
from graph.sufficiency import check_sufficiency
from graph.utils import chat_context_prefix, throttle_llm_if_needed


def retrieve_missing(
    state: GraphState, config: RunnableConfig | None = None, *, index: SearchIndex
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
    # after a failed answer the answer grader's hint, otherwise the context grader's gap
    hint = state.get("reflection") or state.get("context_gap") or ""
    throttle_llm_if_needed()
    missing_query = invoke_missing_query_chain(
        question, context_str, llm, config=cfg, reflection=hint
    )

    iaea_docs, dk_docs = index.search(missing_query, config=cfg)
    merged, new_docs = merge_unique_documents(existing, iaea_docs + dk_docs)
    trusted_merged = list(trusted) + new_docs

    verdict = check_sufficiency(question, merged, llm, config=cfg)

    out: dict[str, Any] = {
        "documents": merged,
        "trusted_documents": trusted_merged,
        "sufficient_after_missing": (
            verdict.sufficient if verdict is not None else False
        ),
        "context_gap": (
            verdict.gap if verdict is not None else state.get("context_gap") or ""
        ),
        "missing_queries": list(state.get("missing_queries") or []) + [missing_query],
    }
    # Only increment retrieval_count when in the initial flow (not retry-after-generation).
    if (state.get("retry_after_generation_count") or 0) == 0:
        out["retrieval_count"] = (state.get("retrieval_count") or 1) + 1
    return out
