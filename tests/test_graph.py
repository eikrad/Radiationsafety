"""Graph routing and node tests."""

from unittest.mock import MagicMock, patch

import pytest

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


def test_decide_after_retrieve_missing_routes_to_web_search_when_not_sufficient():
    """When sufficient_after_missing is False, route to WEB_SEARCH."""
    from graph.graph import decide_after_retrieve_missing

    state: GraphState = {
        "question": "test",
        "documents": [],
        "sufficient_after_missing": False,
    }
    assert decide_after_retrieve_missing(state) == "web_search"


def test_grade_generation_grounded_useful_when_grounded_and_answers(monkeypatch):
    """When generation grader says grounded and answers_question, return 'useful'."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    from graph.graph import grade_generation_grounded

    mock_grader = MagicMock()
    mock_grader.invoke.return_value = MagicMock(grounded=True, answers_question=True)

    def make_grader(_llm=None):
        return mock_grader

    with patch("graph.graph.get_generation_grader", make_grader):
        state: GraphState = {
            "question": "What is radiation?",
            "generation": "Radiation is...",
            "web_search": False,
            "documents": [MagicMock(page_content="doc1")],
            "web_search_attempted": False,
        }
        result = grade_generation_grounded(state)
    assert result == "useful"


def test_grade_generation_grounded_end_when_hallucination(monkeypatch):
    """When generation grader says not grounded, return 'end' (or 'web_search' if enabled)."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    from graph.graph import grade_generation_grounded

    mock_grader = MagicMock()
    mock_grader.invoke.return_value = MagicMock(grounded=False, answers_question=False)

    def make_grader(_llm=None):
        return mock_grader

    with patch("graph.graph.get_generation_grader", make_grader):
        state: GraphState = {
            "question": "What is radiation?",
            "generation": "Random stuff",
            "web_search": False,
            "documents": [MagicMock(page_content="doc1")],
            "web_search_attempted": False,
        }
        result = grade_generation_grounded(state)
    assert result == "end"
