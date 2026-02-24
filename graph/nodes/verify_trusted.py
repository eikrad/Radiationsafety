"""Verify that the generated answer is supported by trusted sources (vector DB + optional trusted web)."""

from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.hallucinations_grader import get_hallucination_grader
from graph.chains.truncate import truncate_docs_for_grader
from graph.i18n import (
    detect_language,
    get_warning_no_trusted_sources,
    get_warning_not_verified_after_web,
    get_warning_not_verified_trusted_only,
)
from graph.llm_factory import get_llm
from graph.nodes.web_search import run_trusted_only_search
from graph.state import GraphState


def verify_trusted(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Check if generation is supported by trusted_documents; optionally try trusted-only web search and re-check."""
    generation = state.get("generation") or ""
    trusted_docs = list(state.get("trusted_documents") or [])
    web_search_attempted = state.get("web_search_attempted", False)
    question = state.get("question") or ""
    cfg = config or {}
    llm = state.get("llm") or get_llm()
    grader = get_hallucination_grader(llm)

    if not trusted_docs:
        lang = detect_language(question)
        return {"retrieval_warning": get_warning_no_trusted_sources(lang)}

    def is_supported(docs) -> bool:
        if not docs:
            return False
        ctx = truncate_docs_for_grader(docs)
        if not ctx.strip():
            return False
        score = grader.invoke(
            {"documents": ctx, "generation": generation},
            config=cfg,
        )
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
