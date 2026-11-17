"""Web search: the seam the graph searches the web through, and Brave behind it.

Only a graph built with web mode "on" is given a WebSearch (graph.graph.build_graph);
the graphs for "off" and privacy mode have none, so they cannot reach the web.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class WebResult:
    link: str
    snippet: str


class WebSearch(Protocol):
    def search(self, query: str, count: int = 5) -> list[WebResult]:
        """The results for `query`; raises when the search itself failed."""
        ...


class BraveWebSearch:
    """Brave Search with BRAVE_SEARCH_API_KEY; without a key it finds nothing."""

    def search(self, query: str, count: int = 5) -> list[WebResult]:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return []
        from langchain_community.tools import BraveSearch

        tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": count})
        return [
            WebResult(link=r["link"], snippet=r["snippet"])
            for r in _parse_brave_results(tool.invoke(query))
            if r["link"] or r["snippet"]
        ]


def _parse_brave_results(results: Any) -> list[dict[str, str]]:
    """Parse BraveSearch result (LangChain returns JSON list of {title, link, snippet}). Returns list of dicts with title, link, snippet."""
    if isinstance(results, str):
        try:
            data = json.loads(results)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [
                {
                    "title": r.get("title") or "",
                    "link": r.get("link") or r.get("url") or "",
                    "snippet": r.get("snippet")
                    or r.get("description")
                    or r.get("title")
                    or "",
                }
                for r in data
                if isinstance(r, dict)
            ]
        if isinstance(data, dict) and "results" in data:
            return _parse_brave_results(data["results"])
        return []
    if isinstance(results, list):
        return [
            {
                "title": r.get("title") or "",
                "link": r.get("link") or r.get("url") or "",
                "snippet": r.get("snippet")
                or r.get("description")
                or r.get("title")
                or "",
            }
            for r in results
            if isinstance(r, dict)
        ]
    return []
