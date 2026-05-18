"""Tests for the Reflexion retry loop: reflection hint flows from grader into retrieve_missing."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from graph.state import GraphState

# ---------------------------------------------------------------------------
# missing_query_chain: reflection parameter
# ---------------------------------------------------------------------------


def test_invoke_missing_query_chain_without_reflection_uses_question_and_context():
    """Without reflection, the chain is called with just question and context."""
    from graph.chains.missing_query_chain import invoke_missing_query_chain

    with patch("graph.chains.missing_query_chain.get_missing_query_chain") as mock_get:
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="dose limits GSR-3")
        mock_get.return_value = chain

        result = invoke_missing_query_chain("What are dose limits?", "Some context.")

    assert result == "dose limits GSR-3"
    called_input = chain.invoke.call_args[0][0]
    assert "reflection" not in called_input or called_input.get("reflection", "") == ""


def test_invoke_missing_query_chain_with_reflection_includes_hint():
    """When reflection is provided, it is included in the chain input."""
    from graph.chains.missing_query_chain import invoke_missing_query_chain

    with patch("graph.chains.missing_query_chain.get_missing_query_chain") as mock_get:
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Annex 2 occupational dose table")
        mock_get.return_value = chain

        result = invoke_missing_query_chain(
            "What are dose limits?",
            "Some context.",
            reflection="occupational dose limits table Annex 2 GSR-3",
        )

    assert result == "Annex 2 occupational dose table"
    called_input = chain.invoke.call_args[0][0]
    # The reflection text is injected into the reflection_hint template variable
    assert "occupational dose limits table Annex 2 GSR-3" in called_input.get(
        "reflection_hint", ""
    )


# ---------------------------------------------------------------------------
# retrieve_missing: passes reflection from state
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> GraphState:
    state: GraphState = {
        "question": "What are dose limits?",
        "generation": "Some answer.",
        "documents": [Document(page_content="existing doc")],
        "trusted_documents": [],
        "web_search": False,
        "web_search_attempted": False,
        "chat_history": [],
        "retrieval_count": 1,
        "retry_after_generation_count": 0,
    }
    state.update(overrides)
    return state


def _patch_retrieve_missing_deps(missing_query: str = "query"):
    """Patch out all external calls in retrieve_missing."""
    return [
        patch(
            "graph.nodes.retrieve_missing.invoke_missing_query_chain",
            return_value=missing_query,
        ),
        patch(
            "graph.nodes.retrieve_missing.invoke_dual_retrievers",
            return_value=([], []),
        ),
        patch(
            "graph.nodes.retrieve_missing.get_context_sufficiency_grader",
            return_value=MagicMock(
                invoke=MagicMock(return_value=MagicMock(binary_score=False))
            ),
        ),
        patch("graph.nodes.retrieve_missing.throttle_llm_if_needed"),
        patch("graph.nodes.retrieve_missing.get_llm", return_value=MagicMock()),
    ]


def test_retrieve_missing_passes_empty_reflection_when_state_has_none():
    """When state has no reflection, invoke_missing_query_chain receives reflection=''."""
    from graph.nodes.retrieve_missing import retrieve_missing

    mock_invoke = MagicMock(return_value="some query")
    patches = _patch_retrieve_missing_deps()
    patches[0] = patch(
        "graph.nodes.retrieve_missing.invoke_missing_query_chain", mock_invoke
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        retrieve_missing(_base_state())

    _, kwargs = mock_invoke.call_args
    assert kwargs.get("reflection", "") == ""


def test_retrieve_missing_passes_reflection_from_state():
    """When state has a reflection hint, it is forwarded to invoke_missing_query_chain."""
    from graph.nodes.retrieve_missing import retrieve_missing

    mock_invoke = MagicMock(return_value="Annex 2 dose table query")
    patches = _patch_retrieve_missing_deps()
    patches[0] = patch(
        "graph.nodes.retrieve_missing.invoke_missing_query_chain", mock_invoke
    )

    state = _base_state(reflection="occupational dose limits table Annex 2 GSR-3")

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        retrieve_missing(state)

    _, kwargs = mock_invoke.call_args
    assert kwargs.get("reflection") == "occupational dose limits table Annex 2 GSR-3"
