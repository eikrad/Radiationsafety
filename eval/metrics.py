"""RAGAS-style evaluation metrics: faithfulness, answer relevance, context precision, context recall."""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.generation_grader import get_generation_grader
from graph.chains.truncate import (
    MAX_CHARS_PER_DOC_GENERATION_GRADER,
    MAX_CONTEXT_CHARS_GENERATION_GRADER,
    truncate_docs_for_grader,
)
from graph.llm_factory import get_llm

# Max chunks to send to per-chunk precision grader (to stay within token limits)
_MAX_CHUNKS_PRECISION = 10
_CHARS_PER_CHUNK_PRECISION = 350


def _format_doc_for_context(doc: Document) -> str:
    """Format one document with [Source: ...] like the generator so the grader sees the same structure."""
    meta = getattr(doc, "metadata", None) or {}
    source = meta.get("source", "retrieved")
    dtype = meta.get("document_type", "")
    label = f"{source} ({dtype})" if dtype else source
    raw = getattr(doc, "page_content", None) or ""
    content = raw[:MAX_CHARS_PER_DOC_GENERATION_GRADER]
    if len(raw) > MAX_CHARS_PER_DOC_GENERATION_GRADER:
        content += "..."
    return f"[Source: {label}]\n{content}"


def _docs_to_context(documents: list[Document]) -> str:
    """Build context string from documents for context-sufficiency grader (default truncation)."""
    return truncate_docs_for_grader(documents) if documents else "No documents"


def _docs_to_context_for_generation_grader(documents: list[Document]) -> str:
    """Build context for generation grader: same format as generator ([Source: ...]) so citations match."""
    if not documents:
        return "No documents"
    parts: list[str] = []
    total = 0
    sep = "\n\n---\n\n"
    sep_len = len(sep)
    for d in documents:
        block = _format_doc_for_context(d)
        need = len(block) + (sep_len if parts else 0)
        if total + need > MAX_CONTEXT_CHARS_GENERATION_GRADER:
            remaining = (
                MAX_CONTEXT_CHARS_GENERATION_GRADER
                - total
                - (sep_len if parts else 0)
                - 20
            )
            if remaining > 0 and block:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total += need
    return sep.join(parts)


def _grader_context(
    documents: list[Document], context_used_for_generation: str = ""
) -> str:
    """Context for generation grader: use exact generator context when available (so eval matches graph)."""
    if (context_used_for_generation or "").strip():
        return context_used_for_generation.strip()
    return _docs_to_context_for_generation_grader(documents)


def faithfulness(
    question: str,
    generation: str,
    documents: list[Document],
    llm: BaseChatModel | None = None,
    context_used_for_generation: str = "",
    **kwargs: Any,
) -> float:
    """Score 0 or 1: is the generation grounded in the retrieved context?"""
    if not generation.strip():
        return 0.0
    grader = get_generation_grader(llm or get_llm())
    ctx = _grader_context(documents, context_used_for_generation)
    result = grader.invoke(
        {"documents": ctx, "question": question, "generation": generation}
    )
    return 1.0 if getattr(result, "grounded", False) else 0.0


def answer_relevance(
    question: str,
    generation: str,
    documents: list[Document],
    llm: BaseChatModel | None = None,
    context_used_for_generation: str = "",
    **kwargs: Any,
) -> float:
    """Score 0 or 1: does the generation address the question?"""
    if not generation.strip():
        return 0.0
    grader = get_generation_grader(llm or get_llm())
    ctx = _grader_context(documents, context_used_for_generation)
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


def context_precision_per_chunk(
    question: str,
    documents: list[Document],
    llm: BaseChatModel | None = None,
    **kwargs: Any,
) -> float:
    """Score 0–1: mean of precision@1, precision@3, precision@5 over chunk relevance (Option A)."""
    if not documents:
        return 0.0
    docs = documents[:_MAX_CHUNKS_PRECISION]
    chunks_str_parts = []
    for i, d in enumerate(docs, 1):
        raw = d.page_content or ""
        text = raw[:_CHARS_PER_CHUNK_PRECISION]
        if len(raw) > _CHARS_PER_CHUNK_PRECISION:
            text += "..."
        chunks_str_parts.append(f"Chunk {i}:\n{text}")
    chunks_str = "\n\n".join(chunks_str_parts)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader. For each chunk below, output 1 if it is relevant to answering the question, 0 otherwise. Reply with a single line of comma-separated 0s and 1s, one per chunk in order (e.g. 1,0,1,0).",
            ),
            ("human", "Question: {question}\n\n{chunks}"),
        ]
    )
    model = llm or get_llm()
    chain = prompt | model
    out = chain.invoke({"question": question, "chunks": chunks_str})
    content = getattr(out, "content", None)
    text = content if isinstance(content, str) else str(out)
    numbers = [int(x.strip()) for x in re.split(r"[\s,]+", text) if x.strip().isdigit()]
    if len(numbers) != len(docs):
        numbers = (
            numbers[: len(docs)]
            if len(numbers) > len(docs)
            else numbers + [0] * (len(docs) - len(numbers))
        )
    if not numbers:
        return 0.0
    k_vals = [k for k in (1, 3, 5) if k <= len(numbers)]
    if not k_vals:
        return float(numbers[0])
    precisions = [sum(numbers[:k]) / k for k in k_vals]
    return sum(precisions) / len(precisions)


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
    use_per_chunk_precision: bool = False,
    context_used_for_generation: str = "",
    **kwargs: Any,
) -> dict[str, float]:
    """Compute faithfulness, answer_relevance, context_precision, context_recall (0–1 each)."""
    precision_fn = (
        context_precision_per_chunk if use_per_chunk_precision else context_precision
    )
    return {
        "faithfulness": faithfulness(
            question,
            generation,
            documents,
            llm=llm,
            context_used_for_generation=context_used_for_generation,
        ),
        "answer_relevance": answer_relevance(
            question,
            generation,
            documents,
            llm=llm,
            context_used_for_generation=context_used_for_generation,
        ),
        "context_precision": precision_fn(question, documents, llm=llm),
        "context_recall": context_recall(
            question, documents, key_facts=key_facts, llm=llm
        ),
    }
