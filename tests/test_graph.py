"""Graph routing and node tests."""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from graph.state import GraphState


def test_decide_to_generate_routes_to_generate_when_the_context_is_sufficient():
    from graph.graph import decide_to_generate

    state: GraphState = {
        "question": "test",
        "generation": "",
        "context_sufficient": True,
        "documents": [],
        "web_search_attempted": False,
    }
    assert decide_to_generate(state) == "generate"


def test_decide_to_generate_searches_the_database_again_when_the_context_is_insufficient():
    """Whatever the web mode (web search used to gate the whole loop)."""
    from graph.graph import decide_to_generate

    state: GraphState = {
        "question": "test",
        "generation": "",
        "context_sufficient": False,
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
    assert decide_after_retrieve_missing(state, web_search_allowed=True) == "generate"


def test_decide_after_retrieve_missing_routes_to_retrieve_missing_again_when_not_sufficient_and_count_under_3():
    """When sufficient_after_missing is False and retrieval_count < 3, route to RETRIEVE_MISSING (third try)."""
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": False,
        "retrieval_count": 2,
    }
    assert (
        decide_after_retrieve_missing(state, web_search_allowed=True)
        == "retrieve_missing"
    )


@pytest.mark.parametrize(
    ("web_search_allowed", "expected"), [(True, "web_search"), (False, "generate")]
)
def test_after_three_insufficient_retrievals_the_web_is_searched_only_if_allowed(
    web_search_allowed, expected
):
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": False,
        "retrieval_count": 3,
    }
    assert (
        decide_after_retrieve_missing(state, web_search_allowed=web_search_allowed)
        == expected
    )


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
    assert decide_after_retrieve_missing(state, web_search_allowed=True) == "generate"


def test_route_after_grade_generation_useful_when_passed():
    """When generation_passed_grading=True, route to 'useful'."""
    web_search_allowed = False
    from graph.graph import route_after_grade_generation

    state: GraphState = {
        "question": "Q",
        "generation": "A",
        "documents": [],
        "web_search_attempted": False,
        "generation_passed_grading": True,
    }
    assert (
        route_after_grade_generation(state, web_search_allowed=web_search_allowed)
        == "useful"
    )


def test_route_after_grade_generation_retries_from_the_database_with_web_search_off():
    web_search_allowed = False
    from graph.graph import route_after_grade_generation

    state: GraphState = {
        "question": "Q",
        "generation": "A",
        "documents": [],
        "web_search_attempted": False,
        "generation_passed_grading": False,
        "retry_after_generation_count": 0,
    }
    assert (
        route_after_grade_generation(state, web_search_allowed=web_search_allowed)
        == "retry_retrieve"
    )


def test_route_after_grade_generation_retry_then_web_search():
    """When grading failed and web search enabled: retry_retrieve up to 2 times, then web_search."""
    web_search_allowed = True
    from graph.graph import route_after_grade_generation

    base: GraphState = {
        "question": "Q",
        "generation": "A",
        "documents": [],
        "web_search_attempted": False,
        "generation_passed_grading": False,
    }
    assert (
        route_after_grade_generation(
            {**base, "retry_after_generation_count": 0},
            web_search_allowed=web_search_allowed,
        )
        == "retry_retrieve"
    )
    assert (
        route_after_grade_generation(
            {**base, "retry_after_generation_count": 1},
            web_search_allowed=web_search_allowed,
        )
        == "retry_retrieve"
    )
    assert (
        route_after_grade_generation(
            {**base, "retry_after_generation_count": 2},
            web_search_allowed=web_search_allowed,
        )
        == "web_search"
    )


def test_route_after_grade_generation_end_when_web_already_attempted():
    """When web search was already attempted and grading still fails, route to 'end'."""
    web_search_allowed = True
    from graph.graph import route_after_grade_generation

    state: GraphState = {
        "question": "Q",
        "generation": "A",
        "documents": [],
        "web_search_attempted": True,
        "generation_passed_grading": False,
        "retry_after_generation_count": 0,
    }
    assert (
        route_after_grade_generation(state, web_search_allowed=web_search_allowed)
        == "end"
    )


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


def test_web_search_deduplicates_existing_results():
    """Web search should not append duplicate result documents."""
    from graph.nodes.web_search import web_search
    from graph.web import WebResult
    from tests.fakes import RecordingWebSearch

    web = RecordingWebSearch(
        [WebResult(link="https://example.org/a", snippet="Duplicate snippet")]
    )
    state: GraphState = {
        "question": "test",
        "generation": "",
        "documents": [
            Document(
                page_content="Duplicate snippet",
                metadata={"source": "https://example.org/a", "document_type": "web"},
            )
        ],
        "web_search_attempted": False,
        "chat_history": [],
    }
    with patch(
        "graph.nodes.web_search.invoke_search_query_chain", return_value="query"
    ):
        out = web_search(state, search=web)
    assert len(out["documents"]) == 1
    assert web.queries == ["query"]


@pytest.mark.parametrize(
    ("allowed", "attempted", "retry_count", "expected"),
    [
        (False, False, 0, "retry_retrieve"),
        (False, False, 2, "end"),
        (True, True, 0, "end"),
        (True, False, 0, "retry_retrieve"),
        (True, False, 1, "retry_retrieve"),
        (True, False, 2, "web_search"),
    ],
)
def test_generation_retry_route_matrix(allowed, attempted, retry_count, expected):
    """Route matrix for retry-after-generation decisions."""
    from graph.graph import _generation_retry_route

    assert (
        _generation_retry_route(
            web_search_allowed=allowed,
            web_search_attempted=attempted,
            retry_count=retry_count,
        )
        == expected
    )


@pytest.mark.parametrize("web_mode", ["off", "private"])
def test_a_graph_without_web_search_has_no_path_to_the_web(web_mode):
    from graph.consts import WEB_SEARCH
    from graph.graph import build_graph
    from tests.fakes import FakeIndex

    graph = build_graph(web_mode, FakeIndex())

    assert WEB_SEARCH not in graph.get_graph().nodes


@pytest.mark.parametrize("web_mode", ["off", "private"])
def test_a_graph_without_web_search_refuses_a_web_search(web_mode):
    from graph.graph import build_graph
    from tests.fakes import FakeIndex, RecordingWebSearch

    with pytest.raises(ValueError, match="web"):
        build_graph(web_mode, FakeIndex(), web_search=RecordingWebSearch())


def test_a_graph_with_web_search_needs_one():
    from graph.consts import WEB_SEARCH
    from graph.graph import build_graph
    from tests.fakes import FakeIndex, RecordingWebSearch

    with pytest.raises(ValueError, match="web"):
        build_graph("on", FakeIndex())
    graph = build_graph("on", FakeIndex(), web_search=RecordingWebSearch())
    assert WEB_SEARCH in graph.get_graph().nodes
