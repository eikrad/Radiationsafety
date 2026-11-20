"""The shared context check, plus each retrieval node's distinct gap policy."""

from importlib import import_module

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve_missing import retrieve_missing
from graph.sufficiency import Sufficiency, check_sufficiency
from tests.fakes import FakeIndex, ScriptedChatModel


class RecordingModel(ScriptedChatModel):
    inputs: list = Field(default_factory=list)
    configs: list = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        delegate = super().with_structured_output(schema, **kwargs)

        def record(prompt, config):
            self.inputs.append(prompt.to_messages()[-1].content)
            self.configs.append(config)
            return delegate.invoke(prompt, config=config)

        return RunnableLambda(record)


FACT = "The deciding fact at the end of the chunk."
DOCUMENT = Document(
    page_content="Introductory text. " * 130 + FACT,
    metadata={"source": "BEK 1384/2025", "document_type": "Danish law"},
)


@pytest.mark.parametrize("sufficient", [True, False])
def test_a_usable_verdict_clears_or_replaces_the_gap(sufficient):
    verdict = check_sufficiency(
        "q", [DOCUMENT], ScriptedChatModel(sufficient=sufficient)
    )

    assert verdict == Sufficiency(
        sufficient,
        "" if sufficient else "Needed: the scripted fact\nFound: not found",
    )


def test_the_check_reads_whole_chunks_with_sources_and_forwards_run_configuration():
    model = RecordingModel()
    web = Document(page_content="Web fact", metadata={"document_type": "web"})

    check_sufficiency(
        "Which rule applies?",
        [DOCUMENT, web],
        model,
        config={"tags": ["sufficiency-test"], "metadata": {"question_id": "q1"}},
    )

    assert len(model.inputs) == 1
    context = model.inputs[0]
    assert "Which rule applies?" in context
    assert "[Source: BEK 1384/2025 (Danish law)]" in context
    assert FACT in context
    assert context.index("Web fact") < context.index(FACT)
    assert "sufficiency-test" in model.configs[0]["tags"]
    assert model.configs[0]["metadata"]["question_id"] == "q1"


def test_the_context_budget_stops_at_the_first_chunk_that_does_not_fit():
    model = RecordingModel()
    oversized = Document(page_content="x" * 60_000)
    later = Document(page_content="later chunk")

    check_sufficiency("q", [DOCUMENT, oversized, later], model)

    assert FACT in model.inputs[0]
    assert "x" * 100 not in model.inputs[0]
    assert "later chunk" not in model.inputs[0]


def test_no_documents_skips_model_construction_and_throttling(monkeypatch):
    def unexpected():
        raise AssertionError("An empty retrieval must not throttle")

    monkeypatch.setattr("graph.sufficiency.throttle_llm_if_needed", unexpected)
    model = BrokenConstructionModel()

    assert check_sufficiency("q", [], model) is None


@pytest.mark.parametrize("text", ["", " " * 10, "x" * 60_000])
def test_nonempty_documents_still_get_checked_when_formatted_context_is_empty(text):
    model = RecordingModel()

    assert check_sufficiency("q", [Document(page_content=text)], model) is not None
    assert len(model.inputs) == 1
    assert model.inputs[0].endswith("Retrieved context:\n\n")


def test_an_invocation_without_a_usable_verdict_returns_no_assessment(monkeypatch):
    pauses = []
    monkeypatch.setattr(
        "graph.sufficiency.throttle_llm_if_needed", lambda: pauses.append(True)
    )
    model = ScriptedChatModel(error=ValueError("No verdict"))

    assert check_sufficiency("q", [DOCUMENT], model) is None
    assert len(model.tracing_seen) == 1
    assert pauses == [True]


def test_unexpected_invocation_errors_propagate():
    error = RuntimeError("Model unavailable")
    with pytest.raises(RuntimeError) as caught:
        check_sufficiency("q", [DOCUMENT], ScriptedChatModel(error=error))
    assert caught.value is error


class BrokenConstructionModel(ScriptedChatModel):
    def with_structured_output(self, schema, **kwargs):
        raise ValueError("Cannot construct the grader")


def test_construction_errors_are_not_mistaken_for_missing_verdicts():
    with pytest.raises(ValueError, match="Cannot construct"):
        check_sufficiency("q", [DOCUMENT], BrokenConstructionModel())


@pytest.mark.parametrize("documents", [[], [DOCUMENT]])
def test_first_check_clears_the_previous_gap_without_a_verdict(documents):
    out = grade_documents(
        {
            "question": "q",
            "documents": documents,
            "context_gap": "previous gap",
            "llm": ScriptedChatModel(error=ValueError("No verdict")),
        }
    )

    assert out == {
        "documents": documents,
        "trusted_documents": documents,
        "context_sufficient": False,
        "context_gap": "",
    }


@pytest.mark.parametrize("documents", [[], [DOCUMENT]])
@pytest.mark.parametrize(
    "sufficient,error", [(True, None), (False, None), (False, ValueError("No verdict"))]
)
def test_later_check_keeps_the_previous_gap_only_without_a_verdict(
    monkeypatch, documents, sufficient, error
):
    monkeypatch.setattr(
        import_module("graph.nodes.retrieve_missing"),
        "invoke_missing_query_chain",
        lambda *args, **kwargs: "follow-up query",
    )
    model = RecordingModel(sufficient=sufficient, error=error)
    out = retrieve_missing(
        {
            "question": "q",
            "documents": documents,
            "trusted_documents": documents,
            "context_gap": "previous gap",
            "llm": model,
        },
        config={"tags": ["later-check"]},
        index=FakeIndex(),
    )

    usable = bool(documents) and error is None
    expected_gap = "" if sufficient else "Needed: the scripted fact\nFound: not found"
    assert out["context_gap"] == (expected_gap if usable else "previous gap")
    assert out["sufficient_after_missing"] is (sufficient if usable else False)
    assert out["documents"] == documents
    assert out["trusted_documents"] == documents
    assert out["missing_queries"] == ["follow-up query"]
    if documents:
        # Existing chunks are rechecked even when the search adds nothing.
        assert len(model.inputs) == 1
        assert "later-check" in model.configs[0]["tags"]
