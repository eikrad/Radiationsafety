"""What eval keeps of an answer, and saving it per run.

graph.answering.Answerer answers each question and its answer trace holds
what the final state has already overwritten:

- initial_documents: output of the first RETRIEVE (before retrieve_missing),
- sufficient: the first GRADE_DOCUMENTS verdict (context_sufficient),
- node_path: every node in execution order (retries and web search are visible),
- missing_queries: each follow-up query retrieve_missing sent to the database.

Full outputs are saved per run (eval/reports/outputs_<run_id>.json, gitignored)
so a run can be re-scored later without re-running the graph.
"""

import gzip
import json
from pathlib import Path

from langchain_core.documents import Document

from graph.answering import Answer

# ranked_*: retrieval-only runs, each collection's deep retrieval in rank order
_DOCUMENT_FIELDS = ("initial_documents", "documents", "ranked_iaea", "ranked_dk")


def answer_output(answer: Answer) -> dict:
    """The record eval scores and saves for one answered question."""
    trace = answer.trace
    return {
        "generation": answer.text,
        "documents": list(answer.documents),
        "context_used_for_generation": trace.context,
        "retrieval_warning": answer.warning,
        "web_search_attempted": answer.used_web_search,
        "initial_documents": list(trace.initial_documents),
        "sufficient": trace.sufficient,
        "node_path": list(trace.node_path),
        "missing_queries": list(trace.missing_queries),
    }


def save_outputs(outputs: dict[str, dict], path: Path) -> None:
    """Write {item_id: answer_output record} as JSON (documents with metadata)."""
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
    """Inverse of save_outputs; also reads an archived, gzipped copy (.json.gz)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
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
