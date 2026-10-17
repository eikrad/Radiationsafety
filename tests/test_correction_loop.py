"""Behaviour of the compiled graph's correction loop, with every model call faked.

The loop (search the database again when the context is insufficient, retry
after a failed answer) was gated on WEB_SEARCH_ENABLED, so with web search off
it never ran: 251 of 251 eval questions took the straight path (ROADMAP, "The
local correction loop never ran"). These tests run the real graph.graph.app.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from eval.graph_run import run_graph
from graph.chains.context_sufficiency_grader import GradeSufficiency
from graph.chains.generation_grader import GradeGeneration, PartVerdict
from graph.chains.hallucinations_grader import GradeHallucinations
from graph.consts import (
    GENERATE,
    GRADE_GENERATION,
    PREPARE_RETRY_RETRIEVE,
    RETRIEVE_MISSING,
    WEB_SEARCH,
)

FIRST = Document(
    page_content="[BEK 1385] Stk. 1. Lukkede kilder skal registreres.",
    metadata={"source": "BEK-1385.xml", "document_type": "dk-law"},
)
MISSING = Document(
    page_content="[BEK 1385] Bilag 6. Sikringsgruppe A gælder for Cs-137 over 100 TBq.",
    metadata={"source": "BEK-1385.xml", "document_type": "dk-law"},
)
GAP = "Bilag 6 sikringsgruppe for Cs-137"


def _chain(*replies):
    chain = MagicMock()
    chain.invoke.side_effect = list(replies)
    return chain


def _sufficiency(*verdicts: bool):
    """The context grader's verdicts in call order (grade_documents, then each retrieve_missing)."""
    return _chain(
        *(
            GradeSufficiency(
                needed=GAP,
                found="Stk. 1 found; Bilag 6 not found",
                binary_score=v,
            )
            for v in verdicts
        )
    )


def _answer_grades(*passes: bool):
    supported = PartVerdict(
        part="registration", status="answered_supported", quote="skal registreres"
    )
    missing = PartVerdict(part="security group", status="not_answered")
    return _chain(
        *(
            GradeGeneration(
                parts=[supported] if ok else [supported, missing],
                missing_info="" if ok else "Bilag 6 sikringsgrupper",
            )
            for ok in passes
        )
    )


def _run(
    *,
    sufficiency: list[bool],
    answers_pass: list[bool],
    privacy_mode: bool = False,
):
    """Run the real graph with fakes; return (captured output, fakes)."""
    suff = _sufficiency(*sufficiency)
    query_chain = MagicMock(side_effect=lambda question, *_, **__: "follow-up query")
    brave = MagicMock()
    brave.from_api_key.return_value.invoke.return_value = "[]"
    fakes = {
        "graph.nodes.retrieve.check_embedding_collections_ready": MagicMock(
            return_value=(True, "")
        ),
        "graph.nodes.retrieve.invoke_dual_retrievers": MagicMock(
            return_value=([], [FIRST])
        ),
        "graph.nodes.retrieve_missing.invoke_dual_retrievers": MagicMock(
            return_value=([], [MISSING])
        ),
        "graph.nodes.retrieve_missing.invoke_missing_query_chain": query_chain,
        "graph.nodes.grade_documents.get_context_sufficiency_grader": MagicMock(
            return_value=suff
        ),
        "graph.nodes.retrieve_missing.get_context_sufficiency_grader": MagicMock(
            return_value=suff
        ),
        "graph.nodes.generate.get_generation_chain": MagicMock(
            return_value=MagicMock(
                invoke=MagicMock(return_value="Lukkede kilder skal registreres.")
            )
        ),
        "graph.nodes.grade_generation.get_generation_grader": MagicMock(
            return_value=_answer_grades(*answers_pass)
        ),
        "graph.nodes.verify_trusted.get_hallucination_grader": MagicMock(
            return_value=MagicMock(
                invoke=MagicMock(return_value=GradeHallucinations(binary_score=True))
            )
        ),
        "graph.nodes.verify_trusted.run_trusted_only_search": MagicMock(
            return_value=""
        ),
        "graph.nodes.web_search.invoke_search_query_chain": MagicMock(
            return_value="web query"
        ),
        "graph.nodes.web_search.BraveSearch": brave,
    }
    from graph.graph import app

    with ExitStack() as stack:
        for target, fake in fakes.items():
            stack.enter_context(patch(target, fake))
        out = run_graph(
            "Hvilken sikringsgruppe gælder for Cs-137?",
            app,
            llm=MagicMock(),
            embedding_provider="scaleway",
            privacy_mode=privacy_mode,
        )
    return out, {"query_chain": query_chain, "web": brave.from_api_key}


@pytest.fixture
def web_off(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")


def test_insufficient_context_is_searched_again_in_the_database_with_web_search_off(
    web_off,
):
    out, fakes = _run(sufficiency=[False, True], answers_pass=[True])

    assert out["node_path"].count(RETRIEVE_MISSING) == 1
    assert WEB_SEARCH not in out["node_path"]
    assert MISSING.page_content in out["context_used_for_generation"]


def test_after_three_database_searches_the_answer_is_written_from_what_was_found(
    web_off,
):
    out, fakes = _run(sufficiency=[False, False, False], answers_pass=[True])

    assert out["node_path"].count(RETRIEVE_MISSING) == 2
    assert WEB_SEARCH not in out["node_path"]
    assert out["generation"]
    fakes["web"].assert_not_called()


def test_a_failed_answer_is_retried_after_another_database_search_with_web_search_off(
    web_off,
):
    out, _ = _run(sufficiency=[True, True], answers_pass=[False, True])

    path = out["node_path"]
    assert path.count(PREPARE_RETRY_RETRIEVE) == 1
    assert path.count(GENERATE) == 2
    assert path.count(GRADE_GENERATION) == 2
    assert WEB_SEARCH not in path


def test_an_answer_that_keeps_failing_is_shown_after_two_retries(web_off):
    out, fakes = _run(
        sufficiency=[True, True, True], answers_pass=[False, False, False]
    )

    assert out["node_path"].count(PREPARE_RETRY_RETRIEVE) == 2
    assert out["node_path"].count(GENERATE) == 3
    assert out["generation"]
    fakes["web"].assert_not_called()


def test_privacy_mode_searches_the_database_again_but_never_the_web(web_on):
    out, fakes = _run(
        sufficiency=[False, False, False, False, False],
        answers_pass=[False, False, False],
        privacy_mode=True,
    )

    assert out["node_path"].count(RETRIEVE_MISSING) >= 2
    assert WEB_SEARCH not in out["node_path"]
    fakes["web"].assert_not_called()


def test_web_search_stays_the_last_resort_when_it_is_enabled(web_on):
    out, fakes = _run(sufficiency=[False, False, False], answers_pass=[True])

    path = out["node_path"]
    fakes["web"].assert_called_once()
    assert path.count(RETRIEVE_MISSING) == 2
    assert path.index(WEB_SEARCH) > max(
        i for i, node in enumerate(path) if node == RETRIEVE_MISSING
    )


def test_the_follow_up_search_is_told_what_the_context_grader_found_missing(web_off):
    _, fakes = _run(sufficiency=[False, True], answers_pass=[True])

    hint = fakes["query_chain"].call_args.kwargs.get("reflection", "")
    assert GAP in hint
    assert "Bilag 6 not found" in hint


def test_the_follow_up_queries_are_recorded_in_the_run_output(web_off):
    out, _ = _run(sufficiency=[False, False, False], answers_pass=[True])

    assert len(out["missing_queries"]) == 2
