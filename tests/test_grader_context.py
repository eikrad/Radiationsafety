"""Graders see what the generator saw: whole chunks with their sources (#129, #136)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from graph.chains.truncate import format_context

# A ~2500-character Danish law chunk whose deciding sentence sits near its end,
# past the 420 (grade_documents) and 1200 (verify_trusted) characters the
# graders used to see.
LATE_FACT = (
    "må efter meddelelsen til arbejdsgiveren om graviditeten ikke være større end 1 mSv"
)
LONG_CHUNK = Document(
    page_content="§ 15. Indledende tekst. " * 100 + LATE_FACT,
    metadata={"source": "BEK 1384/2025", "document_type": "Danish law"},
)
IAEA_CHUNK = Document(
    page_content="The same broad level of protection as for members of the public.",
    metadata={"source": "GSR Part 3", "document_type": "IAEA"},
)


def _grader(verdict: bool):
    grader = MagicMock()
    grader.invoke.return_value = MagicMock(binary_score=verdict)
    return grader


# --- the shared context format ----------------------------------------------------


def test_the_context_keeps_whole_chunks_and_names_each_source():
    context = format_context([LONG_CHUNK, IAEA_CHUNK])

    assert LATE_FACT in context
    assert "[Source: BEK 1384/2025 (Danish law)]" in context
    assert "[Source: GSR Part 3 (IAEA)]" in context


def test_a_capped_context_drops_whole_chunks_never_the_end_of_one():
    context = format_context([IAEA_CHUNK, LONG_CHUNK], max_context_chars=200)

    assert IAEA_CHUNK.page_content in context
    assert "Indledende tekst" not in context


def test_the_generator_sees_the_same_context_format():
    from graph.nodes.generate import generate

    chain = MagicMock()
    chain.invoke.return_value = "answer"
    with (
        patch("graph.nodes.generate.get_generation_chain", return_value=chain),
        patch("graph.nodes.generate.throttle_llm_if_needed"),
    ):
        out = generate(
            {"question": "q", "documents": [LONG_CHUNK, IAEA_CHUNK], "llm": MagicMock()}
        )

    assert out["context_used_for_generation"] == format_context(
        [LONG_CHUNK, IAEA_CHUNK]
    )


# --- grade_documents and retrieve_missing (#129) -----------------------------------


def test_grade_documents_judges_the_whole_chunks():
    from graph.nodes.grade_documents import grade_documents

    grader = _grader(True)
    with (
        patch(
            "graph.nodes.grade_documents.get_context_sufficiency_grader",
            return_value=grader,
        ),
        patch("graph.nodes.grade_documents.throttle_llm_if_needed"),
    ):
        grade_documents(
            {"question": "q", "documents": [LONG_CHUNK], "llm": MagicMock()}
        )

    context = grader.invoke.call_args[0][0]["context"]
    assert LATE_FACT in context
    assert "Danish law" in context


def test_retrieve_missing_judges_the_whole_chunks():
    from graph.nodes.retrieve_missing import retrieve_missing

    grader = _grader(False)
    with (
        patch(
            "graph.nodes.retrieve_missing.invoke_missing_query_chain", return_value="q2"
        ),
        patch(
            "graph.nodes.retrieve_missing.invoke_dual_retrievers", return_value=([], [])
        ),
        patch(
            "graph.nodes.retrieve_missing.get_context_sufficiency_grader",
            return_value=grader,
        ),
        patch("graph.nodes.retrieve_missing.throttle_llm_if_needed"),
    ):
        retrieve_missing(
            {
                "question": "q",
                "documents": [LONG_CHUNK],
                "llm": MagicMock(),
                "chat_history": [],
            }
        )

    assert LATE_FACT in grader.invoke.call_args[0][0]["context"]


def test_the_sufficiency_prompt_requires_the_danish_rule_for_danish_questions():
    from graph.chains.context_sufficiency_grader import system

    assert "Danish" in system and "IAEA" in system


# --- verify_trusted (#136) ---------------------------------------------------------


def _verify(state, grader):
    from graph.nodes.verify_trusted import verify_trusted

    with (
        patch(
            "graph.nodes.verify_trusted.get_hallucination_grader", return_value=grader
        ),
        patch("graph.nodes.verify_trusted.throttle_llm_if_needed"),
    ):
        return verify_trusted({"question": "q", "llm": MagicMock(), **state})


def test_an_answer_grounded_late_in_a_local_chunk_is_verified():
    """#136 regression: the fact sits after character 1200 of a 2500-character chunk."""
    grader = MagicMock()
    grader.invoke.side_effect = lambda inputs, config=None: MagicMock(
        binary_score=LATE_FACT in inputs["documents"]
    )

    out = _verify(
        {"generation": "Højst 1 mSv.", "trusted_documents": [LONG_CHUNK]}, grader
    )

    assert out == {"trusted_verified": True}


def test_an_answer_that_passed_grading_on_local_sources_is_not_checked_again():
    grader = _grader(False)

    out = _verify(
        {
            "generation": "Højst 1 mSv.",
            "trusted_documents": [LONG_CHUNK],
            "documents": [LONG_CHUNK],
            "generation_passed_grading": True,
            "web_search_attempted": False,
        },
        grader,
    )

    assert out == {"trusted_verified": True}
    grader.invoke.assert_not_called()


def test_an_answer_that_used_web_results_is_still_checked_against_trusted_sources():
    grader = _grader(True)
    web = Document(page_content="blog post", metadata={"document_type": "web"})

    _verify(
        {
            "generation": "Højst 1 mSv.",
            "trusted_documents": [LONG_CHUNK],
            "documents": [LONG_CHUNK, web],
            "generation_passed_grading": True,
            "web_search_attempted": True,
        },
        grader,
    )

    grader.invoke.assert_called_once()


def test_an_unsupported_local_answer_still_gets_the_warning():
    out = _verify(
        {"generation": "Højst 5 mSv.", "trusted_documents": [LONG_CHUNK]},
        _grader(False),
    )

    assert "retrieval_warning" in out


def test_the_groundedness_prompt_ignores_framing_and_disclaimers():
    from graph.chains.hallucinations_grader import system

    assert "disclaimer" in system.lower()


def test_the_sufficiency_example_shows_the_reply_as_json():
    """A model that ignores the tool call copies the example's format; it must be JSON
    (a plain 'binary_score: no' line made the reply unparseable)."""
    from graph.chains.context_sufficiency_grader import (
        GradeSufficiency,
        sufficiency_prompt,
    )
    from graph.llm_factory import _parse_text_reply

    rendered = sufficiency_prompt.format_messages(question="q", context="c")[0].content
    example = rendered[rendered.index("Example reply:") :]

    parsed = _parse_text_reply(example, GradeSufficiency)
    assert parsed.binary_score is False
    assert "\nbinary_score:" not in rendered
