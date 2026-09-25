"""Run the RAG graph for one eval question and capture what scoring needs.

graph.invoke only returns the final state, which has already overwritten the
first retrieval and the grade_documents verdict. Streaming node updates keeps
them, without changing the graph itself:

- initial_documents: output of the first RETRIEVE (before retrieve_missing),
- sufficient: the first GRADE_DOCUMENTS verdict (web_search=False means sufficient),
- node_path: every node in execution order (retries and web search are visible).

Full outputs are saved per run (eval/reports/outputs_<run_id>.json, gitignored)
so a run can be re-scored later without re-running the graph.
"""

import json
from pathlib import Path

from langchain_core.documents import Document

from graph.consts import GRADE_DOCUMENTS, RETRIEVE

_DOCUMENT_FIELDS = ("initial_documents", "documents")


def run_graph(
    question: str,
    graph,
    llm=None,
    embedding_provider: str | None = None,
    config: dict | None = None,
) -> dict:
    """Run the graph once and return the final answer plus captured intermediates."""
    graph_input = {
        "question": question,
        "generation": "",
        "web_search": False,
        "documents": [],
        "web_search_attempted": False,
        "chat_history": [],
    }
    if llm is not None:
        graph_input["llm"] = llm
    if embedding_provider is not None:
        graph_input["embedding_provider"] = embedding_provider

    initial_documents = None
    sufficient = None
    node_path: list[str] = []
    final: dict = {}
    for mode, chunk in graph.stream(
        graph_input, config=config or {}, stream_mode=["updates", "values"]
    ):
        if mode == "values":
            final = chunk
            continue
        for node, update in chunk.items():
            node_path.append(node)
            update = update or {}
            if node == RETRIEVE and initial_documents is None:
                initial_documents = list(update.get("documents") or [])
            if node == GRADE_DOCUMENTS and sufficient is None:
                sufficient = not update.get("web_search", False)

    return {
        "generation": final.get("generation", ""),
        "documents": list(final.get("documents") or []),
        "context_used_for_generation": final.get("context_used_for_generation") or "",
        "retrieval_warning": final.get("retrieval_warning"),
        "web_search_attempted": bool(final.get("web_search_attempted", False)),
        "initial_documents": initial_documents or [],
        "sufficient": sufficient,
        "node_path": node_path,
    }


def save_outputs(outputs: dict[str, dict], path: Path) -> None:
    """Write {item_id: run_graph output} as JSON (documents with metadata)."""
    serialisable = {
        item_id: {
            **run,
            **{
                f: [_doc_to_dict(d) for d in run.get(f) or []] for f in _DOCUMENT_FIELDS
            },
        }
        for item_id, run in outputs.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialisable, ensure_ascii=False), encoding="utf-8")


def load_outputs(path: Path) -> dict[str, dict]:
    """Inverse of save_outputs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        item_id: {
            **run,
            **{
                f: [_dict_to_doc(d) for d in run.get(f) or []] for f in _DOCUMENT_FIELDS
            },
        }
        for item_id, run in data.items()
    }


def _doc_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}


def _dict_to_doc(data: dict) -> Document:
    return Document(
        page_content=data.get("page_content", ""), metadata=data.get("metadata", {})
    )
