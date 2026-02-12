"""Generate answer from retrieved context using RAG prompt."""

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer from documents and question."""
    question = state["question"]
    documents = state["documents"]

    context = ""
    if documents:
        context = "\n\n".join(d.page_content for d in documents)

    generation = generation_chain.invoke({"context": context, "question": question})

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }
