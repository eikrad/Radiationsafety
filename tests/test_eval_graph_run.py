"""What eval keeps of an answer: first retrieval, sufficiency verdict, node path."""

from typing import NotRequired, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from eval.graph_run import answer_output, load_outputs, save_outputs
from graph.answering import Answerer, ModelChoice
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, RETRIEVE_MISSING
from tests.fakes import FakeIndex

FIRST = Document(page_content="Første fund", metadata={"source": "BEK-1.xml"})
LATER = Document(page_content="Senere fund", metadata={"source": "GSR-3.pdf"})


class State(TypedDict):
    question: str
    documents: list[Document]
    web_search: bool
    context_sufficient: NotRequired[bool]
    generation: str
    context_used_for_generation: NotRequired[str]
    retrieval_count: NotRequired[int]


def _graph(first_sufficient: bool):
    """Mirrors the real topology: retrieve → grade → (retrieve_missing loop) → generate."""

    def retrieve(state):
        return {"documents": [FIRST], "retrieval_count": 1}

    def grade_documents(state):
        return {
            "web_search": not first_sufficient,
            "context_sufficient": first_sufficient,
        }

    def retrieve_missing(state):
        return {
            "documents": state["documents"] + [LATER],
            "retrieval_count": state["retrieval_count"] + 1,
        }

    def generate(state):
        context = " | ".join(d.page_content for d in state["documents"])
        return {
            "generation": f"Svar: {context}",
            "context_used_for_generation": context,
        }

    def after_grade(state):
        return RETRIEVE_MISSING if state["web_search"] else GENERATE

    def after_missing(state):
        return GENERATE if state["retrieval_count"] >= 3 else RETRIEVE_MISSING

    g = StateGraph(State)
    g.add_node(RETRIEVE, retrieve)
    g.add_node(GRADE_DOCUMENTS, grade_documents)
    g.add_node(RETRIEVE_MISSING, retrieve_missing)
    g.add_node(GENERATE, generate)
    g.add_edge(START, RETRIEVE)
    g.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    g.add_conditional_edges(GRADE_DOCUMENTS, after_grade)
    g.add_conditional_edges(RETRIEVE_MISSING, after_missing)
    g.add_edge(GENERATE, END)
    return g.compile()


def _answerer(graph, built: list | None = None) -> Answerer:
    """An Answerer whose every web mode runs `graph`; records the web modes."""

    def build_graph(web_mode, index, web_search=None):
        if built is not None:
            built.append(web_mode)
        return graph

    return Answerer(
        web_search_enabled=False,
        index_for=lambda embedding_provider: FakeIndex(),
        build_llm=lambda **kwargs: None,
        build_graph=build_graph,
    )


def _recorded(question: str, graph) -> dict:
    """What eval records for one question answered over `graph`."""
    answer = _answerer(graph).answer(question, ModelChoice(provider="scaleway"))
    return answer_output(answer)


def test_the_first_retrieval_is_kept_when_later_retrievals_add_documents():
    run = _recorded("Hvad gælder?", _graph(first_sufficient=False))

    assert [d.page_content for d in run["initial_documents"]] == ["Første fund"]
    assert [d.page_content for d in run["documents"]] == [
        "Første fund",
        "Senere fund",
        "Senere fund",
    ]


def test_the_sufficiency_verdict_of_the_first_grading_is_recorded():
    assert _recorded("q", _graph(first_sufficient=True))["sufficient"] is True
    assert _recorded("q", _graph(first_sufficient=False))["sufficient"] is False


def test_the_node_path_shows_how_often_the_graph_searched_again():
    run = _recorded("q", _graph(first_sufficient=False))

    assert run["node_path"] == [
        RETRIEVE,
        GRADE_DOCUMENTS,
        RETRIEVE_MISSING,
        RETRIEVE_MISSING,
        GENERATE,
    ]


def test_the_answer_and_generator_context_come_from_the_final_state():
    run = _recorded("q", _graph(first_sufficient=True))

    assert run["generation"] == "Svar: Første fund"
    assert run["context_used_for_generation"] == "Første fund"
    assert run["web_search_attempted"] is False


def test_saved_outputs_read_back_with_documents_and_metadata(tmp_path):
    run = _recorded("q", _graph(first_sufficient=False))
    path = tmp_path / "outputs_20260926_101500.json"

    save_outputs({"dk-dose-limits": run}, path)
    restored = load_outputs(path)["dk-dose-limits"]

    assert restored["generation"] == run["generation"]
    assert restored["sufficient"] is False
    assert restored["node_path"] == run["node_path"]
    assert restored["initial_documents"][0].metadata == {"source": "BEK-1.xml"}
    assert [d.page_content for d in restored["documents"]] == [
        d.page_content for d in run["documents"]
    ]


def test_an_eval_in_privacy_mode_runs_the_graph_in_privacy_mode(monkeypatch):
    """The API answers Ollama in privacy mode, on a graph without web search;
    an eval run must do the same, or it measures a mode no user gets."""
    from eval.run_eval import _answer_question, _model_choice

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    built: list[str] = []

    _answer_question(
        "Hvad er dosisgrænsen?",
        _answerer(_graph(first_sufficient=True), built),
        _model_choice(),
    )

    assert built == ["private"]
