"""Fakes for running the real graph end to end without a provider or the web.

Each fake notes whether LangSmith tracing was on at every call, so a test can
show that nothing about a question was traced, worker threads included.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langsmith.utils import tracing_is_enabled
from pydantic import Field

from graph.chains.context_sufficiency_grader import GradeSufficiency
from graph.chains.generation_grader import GradeGeneration, PartVerdict
from graph.chains.hallucinations_grader import GradeHallucinations


class ScriptedChatModel(BaseChatModel):
    """A chat model whose every reply and verdict is fixed in advance.

    sufficient: the context check's verdict; answer_passes: the answer check's;
    grounded: the trusted-source check's.
    error: raised on every call instead of replying.
    """

    reply: str = "Scripted answer."
    sufficient: bool = True
    answer_passes: bool = True
    grounded: bool = True
    error: Exception | None = None
    tracing_seen: list[bool] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _note_call(self) -> None:
        self.tracing_seen.append(bool(tracing_is_enabled()))
        if self.error is not None:
            raise self.error

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._note_call()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.reply))])

    def with_structured_output(self, schema, **kwargs):
        def verdict(_input):
            self._note_call()
            if schema is GradeSufficiency:
                return GradeSufficiency(
                    needed="the scripted fact",
                    found="" if self.sufficient else "not found",
                    binary_score=self.sufficient,
                )
            if schema is GradeGeneration:
                status = "facts_off_topic" if self.answer_passes else "not_answered"
                return GradeGeneration(
                    parts=[PartVerdict(part="the question", status=status)],
                    missing_info="" if self.answer_passes else "the missing fact",
                )
            if schema is GradeHallucinations:
                return GradeHallucinations(binary_score=self.grounded)
            raise AssertionError(f"no scripted verdict for {schema.__name__}")

        return RunnableLambda(verdict)


class RecordingEmbeddings(Embeddings):
    """Fixed-size vectors; notes the tracing state at every call."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.tracing_seen: list[bool] = []

    def embed_documents(self, texts):
        self.tracing_seen.append(bool(tracing_is_enabled()))
        return [[float(len(t))] + [1.0] * (self.dim - 1) for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]
