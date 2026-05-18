"""Graph routing and node tests."""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from graph.state import GraphState


def test_decide_to_generate_routes_to_generate_when_web_search_disabled():
    """Without web_search or WEB_SEARCH_ENABLED, route to GENERATE."""
    from graph.graph import decide_to_generate

    state: GraphState = {
        "question": "test",
        "generation": "",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
    }
    assert decide_to_generate(state) == "generate"


def test_decide_to_generate_routes_to_retrieve_missing_when_enabled(monkeypatch):
    """With web_search=True and WEB_SEARCH_ENABLED, route to RETRIEVE_MISSING (then maybe WEB_SEARCH)."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    from graph.graph import decide_to_generate

    state: GraphState = {
        "question": "test",
        "generation": "",
        "web_search": True,
        "documents": [],
        "web_search_attempted": False,
    }
    assert decide_to_generate(state) == "retrieve_missing"


def test_decide_after_retrieve_missing_routes_to_generate_when_sufficient():
    """When sufficient_after_missing is True, route to GENERATE."""
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": True,
    }
    assert decide_after_retrieve_missing(state) == "generate"


def test_decide_after_retrieve_missing_routes_to_retrieve_missing_again_when_not_sufficient_and_count_under_3():
    """When sufficient_after_missing is False and retrieval_count < 3, route to RETRIEVE_MISSING (third try)."""
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": False,
        "retrieval_count": 2,
    }
    assert decide_after_retrieve_missing(state) == "retrieve_missing"


def test_decide_after_retrieve_missing_routes_to_web_search_when_not_sufficient_and_count_3():
    """When sufficient_after_missing is False and retrieval_count >= 3, route to WEB_SEARCH."""
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": False,
        "retrieval_count": 3,
    }
    assert decide_after_retrieve_missing(state) == "web_search"


def test_decide_after_retrieve_missing_routes_to_generate_when_retry_after_generation():
    """When retry_after_generation_count > 0 (retry path), route to GENERATE."""
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": False,
        "retrieval_count": 3,
        "retry_after_generation_count": 1,
    }
    assert decide_after_retrieve_missing(state) == "generate"


def test_route_after_grade_generation_useful_when_passed(monkeypatch):
    """When generation_passed_grading=True, route to 'useful'."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    from graph.graph import route_after_grade_generation

    state: GraphState = {
        "question": "Q",
        "generation": "A",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
        "generation_passed_grading": True,
    }
    assert route_after_grade_generation(state) == "useful"


def test_route_after_grade_generation_end_when_failed_no_web_search(monkeypatch):
    """When grading failed and web search disabled, route to 'end'."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    from graph.graph import route_after_grade_generation

    state: GraphState = {
        "question": "Q",
        "generation": "A",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
        "generation_passed_grading": False,
        "retry_after_generation_count": 0,
    }
    assert route_after_grade_generation(state) == "end"


def test_route_after_grade_generation_retry_then_web_search(monkeypatch):
    """When grading failed and web search enabled: retry_retrieve up to 2 times, then web_search."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    from graph.graph import route_after_grade_generation

    base: GraphState = {
        "question": "Q",
        "generation": "A",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
        "generation_passed_grading": False,
    }
    assert (
        route_after_grade_generation({**base, "retry_after_generation_count": 0})
        == "retry_retrieve"
    )
    assert (
        route_after_grade_generation({**base, "retry_after_generation_count": 1})
        == "retry_retrieve"
    )
    assert (
        route_after_grade_generation({**base, "retry_after_generation_count": 2})
        == "web_search"
    )


def test_route_after_grade_generation_end_when_web_already_attempted(monkeypatch):
    """When web search was already attempted and grading still fails, route to 'end'."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    from graph.graph import route_after_grade_generation

    state: GraphState = {
        "question": "Q",
        "generation": "A",
        "web_search": False,
        "documents": [],
        "web_search_attempted": True,
        "generation_passed_grading": False,
        "retry_after_generation_count": 0,
    }
    assert route_after_grade_generation(state) == "end"


def test_merge_unique_documents_is_source_aware():
    """Dedupe key should keep same snippet from different sources."""
    from graph.nodes.retrieval_common import merge_unique_documents

    existing = [
        Document(
            page_content="Same content block",
            metadata={"source": "iaea-doc", "document_type": "trusted"},
        )
    ]
    new = [
        Document(
            page_content="Same content block",
            metadata={"source": "dk-law", "document_type": "trusted"},
        ),
        Document(
            page_content="Same content block",
            metadata={"source": "iaea-doc", "document_type": "trusted"},
        ),
    ]
    merged, added = merge_unique_documents(existing, new)
    assert len(merged) == 2
    assert len(added) == 1
    assert merged[1].metadata["source"] == "dk-law"


def test_web_search_deduplicates_existing_results(monkeypatch):
    """Web search should not append duplicate result documents."""
    from graph.nodes.web_search import web_search

    class _FakeTool:
        def invoke(self, _query):
            return [
                {
                    "title": "Existing",
                    "link": "https://example.org/a",
                    "snippet": "Duplicate snippet",
                }
            ]

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    with (
        patch("graph.nodes.web_search.invoke_search_query_chain", return_value="query"),
        patch(
            "graph.nodes.web_search.BraveSearch.from_api_key", return_value=_FakeTool()
        ),
    ):
        state: GraphState = {
            "question": "test",
            "generation": "",
            "web_search": True,
            "documents": [
                Document(
                    page_content="Duplicate snippet",
                    metadata={
                        "source": "https://example.org/a",
                        "document_type": "web",
                    },
                )
            ],
            "web_search_attempted": False,
            "chat_history": [],
        }
        out = web_search(state)
    docs = out["documents"]
    assert len(docs) == 1


@pytest.mark.parametrize(
    ("enabled", "attempted", "retry_count", "expected"),
    [
        (False, False, 0, "end"),
        (True, True, 0, "end"),
        (True, False, 0, "retry_retrieve"),
        (True, False, 1, "retry_retrieve"),
        (True, False, 2, "web_search"),
    ],
)
def test_generation_retry_route_matrix(enabled, attempted, retry_count, expected):
    """Route matrix for retry-after-generation decisions."""
    from graph.graph import _generation_retry_route

    assert (
        _generation_retry_route(
            web_search_enabled=enabled,
            web_search_attempted=attempted,
            retry_count=retry_count,
        )
        == expected
    )
