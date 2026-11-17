"""Web search as the last resort for the context. The query is phrased by the LLM from question + context."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.search_query_chain import invoke_search_query_chain
from graph.consts import env_bool
from graph.nodes.retrieval_common import make_doc_key
from graph.state import GraphState
from graph.utils import chat_context_prefix, throttle_llm_if_needed
from graph.web import WebSearch

# Trusted domains (IAEA + Danish); used when WEB_SEARCH_TRUSTED_DOMAINS_ONLY=true or for verification
DK_DOMAINS = [
    "site:retsinformation.dk",
    "site:sst.dk",
]
IAEA_DOMAIN = "site:iaea.org"


def _build_search_query(base_query: str, trusted_domains_only: bool) -> str:
    """Append domain filter when restricted to trusted sources."""
    q = base_query.strip()
    if trusted_domains_only:
        domain_filter = f" ({IAEA_DOMAIN} OR {' OR '.join(DK_DOMAINS)})"
        return f"{q}{domain_filter}"
    return q


def _trusted_domains_only() -> bool:
    """When True, restrict Brave search to trusted domains (iaea.org, retsinformation.dk, sst.dk)."""
    return env_bool("WEB_SEARCH_TRUSTED_DOMAINS_ONLY")


def web_search(
    state: GraphState, config: RunnableConfig | None = None, *, search: WebSearch
) -> dict[str, Any]:
    """Search the web with a query the LLM phrases from question + current context."""
    question = state["question"]
    existing_docs = list(state.get("documents") or [])
    chat_history = state.get("chat_history") or []
    cfg = config or {}

    doc_context = (
        "\n\n".join(d.page_content[:500] for d in existing_docs)
        if existing_docs
        else "None."
    )
    context = chat_context_prefix(chat_history, max_answer_chars=300) + doc_context
    llm = state.get("llm")
    throttle_llm_if_needed()
    base_query = invoke_search_query_chain(question, context, llm, config=cfg)
    query = _build_search_query(base_query, _trusted_domains_only())

    try:
        results = search.search(query, count=5)
    except Exception:
        # attempted all the same: no second try on a failed search
        return {"documents": existing_docs, "web_search_attempted": True}

    seen_keys = {make_doc_key(d) for d in existing_docs}
    for item in results:
        link = item.link.strip()
        snippet = item.snippet
        if not snippet and not link:
            continue
        # One document per result so sources list shows actual homepages
        source = link if link else "brave_search"
        doc = Document(
            page_content=snippet,
            metadata={"source": source, "document_type": "web", "query": query},
        )
        key = make_doc_key(doc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        existing_docs.append(doc)

    return {"documents": existing_docs, "web_search_attempted": True}


def run_trusted_only_search(
    question: str,
    *,
    search: WebSearch,
    count: int = 5,
    llm=None,
    config: RunnableConfig | None = None,
) -> str:
    """Search only the trusted domains; their snippets joined, or "" if none."""
    cfg = config or {}
    base = invoke_search_query_chain(question, "None.", llm, config=cfg)
    query = _build_search_query(base, trusted_domains_only=True)
    try:
        results = search.search(query, count=count)
    except Exception:
        return ""
    return "\n\n".join(r.snippet for r in results if r.snippet)
