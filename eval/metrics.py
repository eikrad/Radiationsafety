"""RAGAS-style evaluation metrics: faithfulness, answer relevance, context precision, context recall."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.generation_grader import get_generation_grader
from graph.chains.truncate import truncate_docs_for_grader
from graph.llm_factory import get_llm


def _docs_to_context(documents: list[Document]) -> str:
    """Build context string from documents for graders."""
    return truncate_docs_for_grader(documents) if documents else "No documents"


def faithfulness(
    question: str,
    generation: str,
    documents: list[Document],
    llm: BaseChatModel | None = None,
    **kwargs: Any,
) -> float:
    """Score 0 or 1: is the generation grounded in the retrieved context?"""
    if not generation.strip():
        return 0.0
    grader = get_generation_grader(llm or get_llm())
    ctx = _docs_to_context(documents)
    result = grader.invoke(
        {"documents": ctx, "question": question, "generation": generation}
    )
    return 1.0 if getattr(result, "grounded", False) else 0.0


def answer_relevance(
    question: str,
    generation: str,
    documents: list[Document],
    llm: BaseChatModel | None = None,
    **kwargs: Any,
) -> float:
    """Score 0 or 1: does the generation address the question?"""
    if not generation.strip():
        return 0.0
    grader = get_generation_grader(llm or get_llm())
    ctx = _docs_to_context(documents)
    result = grader.invoke(
        {"documents": ctx, "question": question, "generation": generation}
    )
    return 1.0 if getattr(result, "answers_question", False) else 0.0


def context_precision(
    question: str,
    documents: list[Document],
    llm: BaseChatModel | None = None,
    **kwargs: Any,
) -> float:
    """Score 0 or 1: is the top retrieved context sufficient (Option B – one call per question)?"""
    if not documents:
        return 0.0
    grader = get_context_sufficiency_grader(llm or get_llm())
    ctx = _docs_to_context(documents)
    result = grader.invoke({"question": question, "context": ctx})
    return 1.0 if getattr(result, "binary_score", False) else 0.0


def context_recall(
    question: str,
    documents: list[Document],
    key_facts: list[str] | None = None,
    llm: BaseChatModel | None = None,
    **kwargs: Any,
) -> float:
    """Score 0–1: fraction of key_facts present in context, or sufficiency as proxy if no key_facts."""
    if key_facts is not None:
        if len(key_facts) == 0:
            return 0.0
        context_str = _docs_to_context(documents)
        context_lower = context_str.lower()
        found = sum(1 for fact in key_facts if fact.lower() in context_lower)
        return found / len(key_facts)
    if not documents:
        return 0.0
    grader = get_context_sufficiency_grader(llm or get_llm())
    ctx = _docs_to_context(documents)
    result = grader.invoke({"question": question, "context": ctx})
    return 1.0 if getattr(result, "binary_score", False) else 0.0


def compute_all_metrics(
    question: str,
    generation: str,
    documents: list[Document],
    expected_answer: str | None = None,
    key_facts: list[str] | None = None,
    llm: BaseChatModel | None = None,
) -> dict[str, float]:
    """Compute faithfulness, answer_relevance, context_precision, context_recall (0–1 each)."""
    return {
        "faithfulness": faithfulness(question, generation, documents, llm=llm),
        "answer_relevance": answer_relevance(
            question, generation, documents, llm=llm
        ),
        "context_precision": context_precision(question, documents, llm=llm),
        "context_recall": context_recall(
            question, documents, key_facts=key_facts, llm=llm
        ),
    }
