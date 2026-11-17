"""Answer one question: the single way the API and eval run the graph.

Privacy mode follows from the model choice here and nowhere else, and so does
the rule that a question is never traced in privacy mode or when the user
brought their own API keys.
"""

import os
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import langsmith as ls
from langchain_core.documents import Document

from graph.consts import GRADE_DOCUMENTS, RETRIEVE
from graph.llm_factory import (
    ALLOWED_PROVIDERS,
    DEFAULT_PROVIDER,
    APIKeyError,
    ProviderConfigError,
    get_embedding_provider,
    get_llm,
)

_RUN_CONFIG = {"run_name": "RadiationSafetyRAG", "tags": ["rag", "radiation-safety"]}


class RateLimited(Exception):
    """The provider refused the request: rate limit or quota."""

    def __init__(self, provider: str):
        super().__init__(f"{provider} rate limit or quota exceeded")
        self.provider = provider


class TimedOut(Exception):
    """The provider did not answer within its time limit."""

    def __init__(self, provider: str):
        super().__init__(f"{provider} did not answer in time")
        self.provider = provider


class LocalModelUnavailable(Exception):
    """Privacy mode could not reach the local model; the message says how to fix it."""


@dataclass(frozen=True)
class ModelChoice:
    """The answering model a request asks for, with any keys the user brought.

    provider: None (or one not offered) means the server's LLM_PROVIDER.
    """

    provider: str | None = None
    api_keys: Mapping[str, str] = field(default_factory=dict)
    variant: str | None = None

    def resolved_provider(self) -> str:
        provider = (
            self.provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
        ).strip()
        provider = provider.lower()
        return provider if provider in ALLOWED_PROVIDERS else DEFAULT_PROVIDER

    @property
    def privacy_mode(self) -> bool:
        """The local model: nothing about the question may leave the machine."""
        return self.resolved_provider() == "ollama"


@dataclass(frozen=True)
class AnswerTrace:
    """How an answer came about; eval scores it, users never see it."""

    initial_documents: list[Document]
    sufficient: bool | None
    node_path: list[str]
    missing_queries: list[str]
    context: str


@dataclass(frozen=True)
class Answer:
    text: str
    documents: list[Document]
    warning: str | None
    used_web_search: bool
    privacy_mode: bool
    routing_outcome: str | None
    chat_history: list[tuple[str, str]]
    trace: AnswerTrace

    @property
    def sources(self) -> list[tuple[str, str | None]]:
        """(source, document type) of the documents, each once, in order."""
        seen: list[tuple[str, str | None]] = []
        for doc in self.documents:
            meta = doc.metadata or {}
            key = (str(meta.get("source", "retrieved")), meta.get("document_type"))
            if key not in seen:
                seen.append(key)
        return seen


class Answerer:
    """Runs the graph for one question at a time.

    graph: the compiled graph (default: graph.graph.app).
    build_llm: builds the answering model from provider, api_key, model_variant.
    run_config: the LangSmith run name and tags, when a question is traced.
    """

    def __init__(
        self,
        graph: Any = None,
        *,
        build_llm: Callable[..., Any] = get_llm,
        run_config: dict | None = None,
    ):
        self._graph = graph
        self._build_llm = build_llm
        self._run_config = run_config or _RUN_CONFIG

    def _compiled_graph(self):
        if self._graph is None:
            from graph.graph import app

            self._graph = app
        return self._graph

    def answer(
        self,
        question: str,
        choice: ModelChoice | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> Answer:
        """Raises APIKeyError or ProviderConfigError before anything runs, and
        RateLimited, TimedOut or LocalModelUnavailable when the model fails."""
        choice = choice or ModelChoice()
        provider = choice.resolved_provider()
        try:
            return self._answer(question, choice, provider, history)
        except (APIKeyError, ProviderConfigError):
            raise
        except Exception as e:
            failure = _named_failure(e, provider, choice.variant)
            if failure is None:
                raise
            raise failure from e

    def _answer(
        self,
        question: str,
        choice: ModelChoice,
        provider: str,
        history: list[tuple[str, str]] | None,
    ) -> Answer:
        privacy_mode = choice.privacy_mode
        llm = self._build_llm(
            provider=provider,
            api_key=choice.api_keys.get(provider) or None,
            model_variant=choice.variant,
        )
        graph_input = {
            "question": question,
            "generation": "",
            "web_search": False,
            "documents": [],
            "web_search_attempted": False,
            "chat_history": list(history or []),
            "llm": llm,
            "embedding_provider": get_embedding_provider(provider),
            "privacy_mode": privacy_mode,
        }
        # never traced in privacy mode or with the user's own keys; otherwise
        # whatever the server (or an enclosing tracing_context) says
        untraced = privacy_mode or bool(choice.api_keys)
        with ls.tracing_context(enabled=False) if untraced else nullcontext():
            final, trace = self._run(graph_input)
        return Answer(
            text=final.get("generation", ""),
            documents=list(final.get("documents") or []),
            warning=final.get("retrieval_warning"),
            used_web_search=bool(final.get("web_search_attempted", False)),
            privacy_mode=privacy_mode,
            routing_outcome=final.get("routing_outcome"),
            chat_history=list(final.get("chat_history") or history or []),
            trace=trace,
        )

    def _run(self, graph_input: dict) -> tuple[dict, AnswerTrace]:
        """Stream the graph: the final state has overwritten the first retrieval
        and the first sufficiency verdict, the node updates still have them."""
        initial_documents = None
        sufficient = None
        node_path: list[str] = []
        final: dict = {}
        for mode, chunk in self._compiled_graph().stream(
            graph_input, config=self._run_config, stream_mode=["updates", "values"]
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
                    sufficient = update.get("context_sufficient")
        trace = AnswerTrace(
            initial_documents=initial_documents or [],
            sufficient=sufficient,
            node_path=node_path,
            missing_queries=list(final.get("missing_queries") or []),
            context=final.get("context_used_for_generation") or "",
        )
        return final, trace


def _named_failure(
    exc: Exception, provider: str, model_variant: str | None
) -> Exception | None:
    """The error a caller can act on, or None for an unexpected one."""
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return RateLimited(provider)
    if _is_timeout(exc):
        return TimedOut(provider)
    if provider == "ollama":
        return LocalModelUnavailable(_local_model_problem(exc, model_variant))
    return None


def _is_timeout(exc: BaseException) -> bool:
    """Whether a provider call gave up on its time limit (any client library)."""
    import httpx

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if (
            isinstance(current, (TimeoutError, httpx.TimeoutException))
            or ("Timeout" in name and "Error" in name)
            or name == "DeadlineExceeded"  # google-api-core (Gemini)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _local_model_problem(exc: Exception, model_variant: str | None) -> str:
    """What went wrong with the local model, and the command that fixes it."""
    from graph.llm_factory import get_embedding_model_name, ollama_model_name

    err = str(exc).lower()
    llm_model = model_variant or ollama_model_name()
    embed_model = get_embedding_model_name("ollama")
    if "not found" in err or "404" in err:
        if embed_model.lower() in err:
            return (
                f"Ollama embedding model '{embed_model}' is not installed. "
                f"Run: ollama pull {embed_model}"
            )
        return (
            f"Ollama model '{llm_model}' is not installed. Run: ollama pull {llm_model}"
        )
    if "connect" in type(exc).__name__.lower() or "connection refused" in err:
        return "Ollama is not running. Start it with: ollama serve"
    return f"Ollama error: {exc}"
