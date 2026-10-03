"""Generate answer from retrieved context using RAG prompt."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from graph.chains.generation import get_generation_chain
from graph.chains.truncate import format_context
from graph.llm_factory import get_llm
from graph.state import GraphState
from graph.utils import throttle_llm_if_needed


def _format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format chat history for the prompt."""
    if not history:
        return ""
    lines = []
    for q, a in history:
        lines.append(f"User: {q}\nAssistant: {a}")
    return "\n\n".join(lines) + "\n\n" if lines else ""


def generate(state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Generate answer from documents, question, and optional chat history."""
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history") or []
    cfg = config or {}
    llm = state.get("llm") or get_llm()
    throttle_llm_if_needed()
    chain = get_generation_chain(llm)

    context = format_context(documents) if documents else ""

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
        "context_used_for_generation": context,
        "chat_history": updated_history,
        "reflection": "",  # clear stale Reflexion hint before GRADE_GENERATION runs
    }
