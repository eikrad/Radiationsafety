"""Generate answer from retrieved context using RAG prompt."""

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

from graph.chains.generation import get_generation_chain
from graph.llm_factory import get_llm
from graph.state import GraphState


def _format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format chat history for the prompt."""
    if not history:
        return ""
    lines = []
    for q, a in history:
        lines.append(f"User: {q}\nAssistant: {a}")
    return "\n\n".join(lines) + "\n\n" if lines else ""


def _format_document(doc: Any) -> str:
    """Format one document with its source so the model can use and cite it (especially web results)."""
    meta = getattr(doc, "metadata", {}) or {}
    source = meta.get("source", "retrieved")
    dtype = meta.get("document_type", "")
    label = source
    if dtype:
        label = f"{source} ({dtype})"
    return f"[Source: {label}]\n{doc.page_content}"


def generate(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Generate answer from documents, question, and optional chat history."""
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history") or []
    cfg = config or {}
    llm = state.get("llm") or get_llm()
    chain = get_generation_chain(llm)

    context = ""
    if documents:
        # Put web results first so the model sees them before long document chunks
        ordered = sorted(documents, key=lambda d: (0 if (getattr(d, "metadata", {}) or {}).get("document_type") == "web" else 1,))
        context = "\n\n---\n\n".join(_format_document(d) for d in ordered)

    chat_history_str = _format_chat_history(chat_history)
    generation = chain.invoke(
        {
            "context": context,
            "chat_history": chat_history_str,
            "question": question,
        },
        config=cfg,
    )

    updated_history = list(chat_history) + [(question, generation)]

    return {
        "generation": generation,
        "chat_history": updated_history,
    }
