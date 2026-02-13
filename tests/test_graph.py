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


def test_decide_to_generate_routes_to_web_search_when_enabled(monkeypatch):
    """With web_search=True and WEB_SEARCH_ENABLED, route to WEB_SEARCH."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    from graph.graph import decide_to_generate

    state: GraphState = {
        "question": "test",
        "generation": "",
        "web_search": True,
        "documents": [],
        "web_search_attempted": False,
    }
    assert decide_to_generate(state) == "web_search"


def test_grade_generation_grounded_useful_when_grounded_and_answers(monkeypatch):
    """When both graders pass, return 'useful'."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    from graph.graph import grade_generation_grounded

    mock_hall = MagicMock()
    mock_hall.invoke.return_value = MagicMock(binary_score=True)
    mock_ans = MagicMock()
    mock_ans.invoke.return_value = MagicMock(binary_score=True)

    with patch("graph.graph.hallucination_grader", mock_hall):
        with patch("graph.graph.answer_grader", mock_ans):
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
    """When hallucination grader fails, return 'end' (or 'web_search' if enabled)."""
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    from graph.graph import grade_generation_grounded

    mock_hall = MagicMock()
    mock_hall.invoke.return_value = MagicMock(binary_score=False)
    mock_ans = MagicMock()

    with patch("graph.graph.hallucination_grader", mock_hall):
        with patch("graph.graph.answer_grader", mock_ans):
            state: GraphState = {
                "question": "What is radiation?",
                "generation": "Random stuff",
                "web_search": False,
                "documents": [MagicMock(page_content="doc1")],
                "web_search_attempted": False,
            }
            result = grade_generation_grounded(state)
    assert result == "end"
