"""Second retrieval targeting 'missing' information; re-check sufficiency before web search."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.missing_query_chain import invoke_missing_query_chain
from graph.chains.truncate import truncate_docs_for_grader
from graph.llm_factory import get_llm
from graph.state import GraphState
from graph.utils import chat_context_prefix
from ingestion import get_retrievers


def retrieve_missing(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Retrieve again with an LLM-generated query for missing info; merge with existing docs; set sufficient_after_missing."""
    question = state["question"]
    existing = list(state.get("documents") or [])
    trusted = list(state.get("trusted_documents") or [])
    chat_history = state.get("chat_history") or []
    cfg = config or {}
    llm = state.get("llm") or get_llm()

    doc_context = "\n\n".join(d.page_content for d in existing) if existing else "None."
    context_str = chat_context_prefix(chat_history) + "Document context:\n" + doc_context
    missing_query = invoke_missing_query_chain(question, context_str, llm, config=cfg)

    iaea_retriever, dk_retriever = get_retrievers()
    def _invoke_iaea():
        return iaea_retriever.invoke(missing_query, config=cfg)

    def _invoke_dk():
        return dk_retriever.invoke(missing_query, config=cfg)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_iaea = executor.submit(_invoke_iaea)
        fut_dk = executor.submit(_invoke_dk)
        iaea_docs = fut_iaea.result()
        dk_docs = fut_dk.result()

    seen = {d.page_content[:200] for d in existing}
    merged = list(existing)
    new_docs: list[Document] = []
    for d in iaea_docs + dk_docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            merged.append(d)
            new_docs.append(d)
    trusted_merged = list(trusted) + new_docs

    sufficient = False
    if merged:
        truncated = truncate_docs_for_grader(merged)
        sufficiency = get_context_sufficiency_grader(llm)
        result = sufficiency.invoke(
            {"question": question, "context": truncated},
            config=cfg,
        )
        sufficient = bool(result.binary_score)

    return {
        "documents": merged,
        "trusted_documents": trusted_merged,
        "sufficient_after_missing": sufficient,
    }
