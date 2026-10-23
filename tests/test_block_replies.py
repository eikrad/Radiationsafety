"""Gemini 3 replies with a list of content blocks, not a string.

The first Gemini 3.8 Flash smoke run failed on every question with
"'list' object has no attribute 'strip'".
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from graph.chains.context_sufficiency_grader import GradeSufficiency


def _blocks(text: str) -> AIMessage:
    return AIMessage(
        content=[{"type": "text", "text": text, "extras": {"signature": "abc"}}]
    )


def test_a_follow_up_query_is_read_from_a_block_reply():
    from graph.chains.missing_query_chain import invoke_missing_query_chain

    with patch("graph.chains.missing_query_chain.get_missing_query_chain") as get:
        get.return_value = MagicMock(
            invoke=MagicMock(return_value=_blocks(" Bilag 6 "))
        )
        assert invoke_missing_query_chain("Q?", "context") == "Bilag 6"


def test_a_web_search_query_is_read_from_a_block_reply():
    from graph.chains.search_query_chain import invoke_search_query_chain

    with patch("graph.chains.search_query_chain.get_search_query_chain") as get:
        get.return_value = MagicMock(invoke=MagicMock(return_value=_blocks("sst.dk")))
        assert invoke_search_query_chain("Q?", "context") == "sst.dk"


def test_a_verdict_written_as_text_in_a_block_reply_is_parsed():
    from graph.llm_factory import with_text_fallback

    reply = _blocks('{"needed": "x", "found": "y", "binary_score": true}')
    structured = RunnableLambda(
        lambda prompt: {"raw": reply, "parsed": None, "parsing_error": None}
    )
    grader = with_text_fallback(structured, GradeSufficiency, include_raw=False)

    assert grader.invoke("prompt").binary_score is True
