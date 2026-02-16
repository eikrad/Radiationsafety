"""Brave Search fallback for IAEA and Danish radiation safety sites."""

import json
import os
from typing import Any, Dict, List

from langchain_community.tools import BraveSearch
from langchain_core.documents import Document

from graph.state import GraphState

# Danish fallback domains from plan
DK_DOMAINS = [
    "site:retsinformation.dk",
    "site:sst.dk",
]
IAEA_DOMAIN = "site:iaea.org"


def _extract_contents(results: Any) -> List[str]:
    """Extract text contents from BraveSearch result (JSON string or list)."""
    if isinstance(results, str):
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            return [results]
        if isinstance(data, list):
            return [
                (r.get("description") or r.get("title") or str(r))
                for r in data
                if isinstance(r, dict)
            ]
        if isinstance(data, dict) and "results" in data:
            return _extract_contents(data["results"])
        return [str(data)]
    if isinstance(results, list):
        return [
            (r.get("description") or r.get("title") or str(r))
            for r in results
            if isinstance(r, dict)
        ]
    return []


def web_search(state: GraphState) -> Dict[str, Any]:
    """Run Brave Search restricted to IAEA and Danish legislation domains."""
    question = state["question"]
    existing_docs = list(state.get("documents") or [])
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return {
            "documents": existing_docs,
            "web_search": False,
            "web_search_attempted": True,  # prevent infinite retry when key missing
        }

    domain_filter = f" ({IAEA_DOMAIN} OR {' OR '.join(DK_DOMAINS)})"
    query = f"{question}{domain_filter}"

    try:
        tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 5})
        results = tool.invoke(query)
    except Exception:
        return {
            "documents": existing_docs,
            "web_search": False,
            "web_search_attempted": True,  # prevent infinite retry on failure
        }

    contents = _extract_contents(results)
    if contents:
        web_doc = Document(
            page_content="\n\n".join(contents),
            metadata={"source": "brave_search", "query": question},
        )
        existing_docs = existing_docs + [web_doc]

    return {
        "documents": existing_docs,
        "web_search": False,
        "web_search_attempted": True,
    }
