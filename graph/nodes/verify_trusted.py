"""Verify that the generated answer is supported by trusted sources (vector DB + optional trusted web)."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.hallucinations_grader import get_hallucination_grader
from graph.chains.truncate import MAX_GRADER_CONTEXT_CHARS, format_context
from graph.i18n import (
    detect_language,
    get_warning_no_trusted_sources,
    get_warning_not_verified_after_web,
    get_warning_not_verified_trusted_only,
)
from graph.llm_factory import get_llm
from graph.nodes.web_search import run_trusted_only_search
from graph.state import GraphState
from graph.utils import throttle_llm_if_needed


def verify_trusted(
    state: GraphState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Check if generation is supported by trusted_documents; optionally try trusted-only web search and re-check."""
    generation = state.get("generation") or ""
    trusted_docs = list(state.get("trusted_documents") or [])
    web_search_attempted = state.get("web_search_attempted", False)
    question = state.get("question") or ""
    cfg = config or {}

    if not trusted_docs:
        lang = detect_language(question)
        return {"retrieval_warning": get_warning_no_trusted_sources(lang)}
    # grade_generation already found the answer grounded in the full context the
    # generator saw; without web results that context is exactly the trusted
    # documents, so a second, differently prompted call can only disagree (#136).
    if state.get("generation_passed_grading") and not web_search_attempted:
        return {"trusted_verified": True}

    llm = state.get("llm") or get_llm()
    grader = get_hallucination_grader(llm)

    def is_supported(docs) -> bool:
        if not docs:
            return False
        ctx = format_context(docs, max_context_chars=MAX_GRADER_CONTEXT_CHARS)
        if not ctx.strip():
            return False
        throttle_llm_if_needed()
        try:
            score = grader.invoke(
                {"documents": ctx, "generation": generation},
                config=cfg,
            )
        except ValueError:  # no verdict even when asked again: not verified
            return False
        return bool(score.binary_score)

    if is_supported(trusted_docs):
        return {"trusted_verified": True}

    if web_search_attempted:
        supplemental = run_trusted_only_search(question, llm=llm, config=cfg)
        if supplemental:
            extra = Document(page_content=supplemental, metadata={})
            if is_supported(trusted_docs + [extra]):
                return {"trusted_verified": True}
        lang = detect_language(question)
        return {"retrieval_warning": get_warning_not_verified_after_web(lang)}

    lang = detect_language(question)
    return {"retrieval_warning": get_warning_not_verified_trusted_only(lang)}
