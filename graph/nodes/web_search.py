"""Brave Search fallback for RAG context. Query is phrased by LLM from question + context."""

import json
import os
from typing import Any, Dict, List, Optional

from langchain_community.tools import BraveSearch
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

from graph.chains.search_query_chain import invoke_search_query_chain
from graph.consts import env_bool
from graph.state import GraphState
from graph.utils import chat_context_prefix

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


def _parse_brave_results(results: Any) -> List[Dict[str, str]]:
    """Parse BraveSearch result (LangChain returns JSON list of {title, link, snippet}). Returns list of dicts with title, link, snippet."""
    if isinstance(results, str):
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [
                {"title": r.get("title") or "", "link": r.get("link") or r.get("url") or "", "snippet": r.get("snippet") or r.get("description") or r.get("title") or ""}
                for r in data
                if isinstance(r, dict)
            ]
        if isinstance(data, dict) and "results" in data:
            return _parse_brave_results(data["results"])
        return []
    if isinstance(results, list):
        return [
            {"title": r.get("title") or "", "link": r.get("link") or r.get("url") or "", "snippet": r.get("snippet") or r.get("description") or r.get("title") or ""}
            for r in results
            if isinstance(r, dict)
        ]
    return []


def web_search(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Run Brave Search. Query is phrased by LLM from question + current context for better results."""
    question = state["question"]
    existing_docs = list(state.get("documents") or [])
    chat_history = state.get("chat_history") or []
    cfg = config or {}
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return {
            "documents": existing_docs,
            "web_search": False,
            "web_search_attempted": True,  # prevent infinite retry when key missing
        }

    doc_context = (
        "\n\n".join(d.page_content[:500] for d in existing_docs)
        if existing_docs
        else "None."
    )
    context = chat_context_prefix(chat_history, max_answer_chars=300) + doc_context
    llm = state.get("llm")
    base_query = invoke_search_query_chain(question, context, llm, config=cfg)
    query = _build_search_query(base_query, _trusted_domains_only())

    try:
        tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 5})
        results = tool.invoke(query)
    except Exception:
        return {
            "documents": existing_docs,
            "web_search": False,
            "web_search_attempted": True,  # prevent infinite retry on failure
        }

    parsed = _parse_brave_results(results)
    for item in parsed:
        link = (item.get("link") or "").strip()
        snippet = item.get("snippet") or item.get("title") or ""
        if not snippet and not link:
            continue
        # One document per result so sources list shows actual homepages
        source = link if link else "brave_search"
        existing_docs.append(
            Document(
                page_content=snippet,
                metadata={"source": source, "document_type": "web", "query": query},
            )
        )

    return {
        "documents": existing_docs,
        "web_search": False,
        "web_search_attempted": True,
    }


def run_trusted_only_search(
    question: str, count: int = 5, llm=None, config: Optional[RunnableConfig] = None
) -> str:
    """Run Brave Search restricted to trusted domains. Returns concatenated snippets or empty string."""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return ""
    cfg = config or {}
    base = invoke_search_query_chain(question, "None.", llm, config=cfg)
    query = _build_search_query(base, trusted_domains_only=True)
    try:
        tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": count})
        results = tool.invoke(query)
        parsed = _parse_brave_results(results)
        snippets = [p.get("snippet") or p.get("title") or "" for p in parsed if p.get("snippet") or p.get("title")]
        return "\n\n".join(snippets) if snippets else ""
    except Exception:
        return ""
