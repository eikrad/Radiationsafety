"""What eval keeps of an answer.

graph.answering.Answerer answers each question and its answer trace holds
what the final state has already overwritten:

- initial_documents: output of the first RETRIEVE (before retrieve_missing),
- sufficient: the first GRADE_DOCUMENTS verdict (context_sufficient),
- node_path: every node in execution order (retries and web search are visible),
- missing_queries: each follow-up query retrieve_missing sent to the database.

eval.saved_runs persists these records so they can be scored again later.
"""

from graph.answering import Answer


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
