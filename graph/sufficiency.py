"""Check whether the retrieved context can fully answer a question.

Both the first retrieval and the database correction loop use this check.
An absent verdict is distinct from an insufficient verdict: the caller may
still have a previous gap worth searching for.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from graph.chains.context_sufficiency_grader import get_context_sufficiency_grader
from graph.chains.truncate import MAX_GRADER_CONTEXT_CHARS, format_context
from graph.utils import throttle_llm_if_needed


@dataclass(frozen=True)
class Sufficiency:
    """A usable verdict; a sufficient context has no gap."""

    sufficient: bool
    gap: str


def check_sufficiency(
    question: str,
    documents: list[Document],
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> Sufficiency | None:
    """Return a verdict, or None for no documents or no usable verdict.

    Only ValueError during invocation or verdict interpretation means no
    verdict. Preparation and model construction errors still propagate.
    The caller owns what to do with a previous gap when no verdict is given.
    """
    if not documents:
        return None

    context = format_context(documents, max_context_chars=MAX_GRADER_CONTEXT_CHARS)
    throttle_llm_if_needed()
    grader = get_context_sufficiency_grader(llm)
    try:
        verdict = grader.invoke(
            {"question": question, "context": context}, config=config or {}
        )
        sufficient = bool(verdict.binary_score)
        gap = "" if sufficient else f"Needed: {verdict.needed}\nFound: {verdict.found}"
        return Sufficiency(sufficient, gap)
    except ValueError:
        return None
