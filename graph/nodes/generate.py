"""Generate answer from retrieved context using RAG prompt."""

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def _format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format chat history for the prompt."""
    if not history:
        return ""
    lines = []
    for q, a in history:
        lines.append(f"User: {q}\nAssistant: {a}")
    return "\n\n".join(lines) + "\n\n" if lines else ""


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer from documents, question, and optional chat history."""
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history") or []

    context = ""
    if documents:
        context = "\n\n".join(d.page_content for d in documents)

    chat_history_str = _format_chat_history(chat_history)
    generation = generation_chain.invoke(
        {
            "context": context,
            "chat_history": chat_history_str,
            "question": question,
        }
    )

    updated_history = list(chat_history) + [(question, generation)]

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "chat_history": updated_history,
    }
