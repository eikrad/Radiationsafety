"""Eval judge: two narrow LLM calls per question.

A) Nugget assignment: which nuggets does the answer contain? Answer + nuggets
   only, no retrieved context, so the call is small. Listwise, at most 10
   nuggets per call, labels support / partial_support / not_support
   (AutoNuggetizer, Pradeep et al. 2025).
B) Groundedness: which claims in the answer does the retrieved context not
   support, and did the answer refuse? Answer + context.

Two focused calls rather than one combined prompt: GroUSE (Muller et al. 2024)
measured a drop from 69 % to 40 % of passed judge unit tests when all
metrics were asked in one prompt. The judge should be a different model than
the one that generated the answer (see eval/README.md).
"""

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_NUGGETS_PER_CALL = 10
_ATTEMPTS = 2  # first try + one retry

Label = Literal["support", "partial_support", "not_support"]


class NuggetLabels(BaseModel):
    """One label per nugget, in the order given."""

    labels: list[Label] = Field(
        description="Exactly one label per nugget, in the same order as the nugget list."
    )


class Groundedness(BaseModel):
    """Unsupported claims in the answer, and whether it refused to answer."""

    unsupported_claims: list[str] = Field(
        description=(
            "Each factual statement in the answer that the context does not support, "
            "briefly quoted or paraphrased. Empty list if everything is supported."
        )
    )
    refused: bool = Field(
        description=(
            "True if the answer declines to answer the question substantively (for "
            "example says the sources do not contain the information), even if it "
            "adds related information. False if it answers."
        )
    )


_NUGGET_SYSTEM = """You assess answers of a radiation-safety question-answering system.

A nugget is an atomic fact that a good answer to the question should contain. \
For each nugget, decide whether the ANSWER contains it:
- support: the answer fully states the fact.
- partial_support: the answer states part of it, e.g. the right concept but a \
missing or vaguer number, unit, condition or legal reference.
- not_support: the answer does not state it, or contradicts it.

Rules:
- Judge meaning, not wording. Answer and nuggets may be in different languages \
(Danish, English); a correct translation counts.
- Numbers, units and legal references must match: Gy is not Sv, mSv is not µSv, \
annex 2 is not annex 3.
- Compare only the answer with the nuggets. Do not use your own knowledge and do \
not judge whether the nugget itself is true.
- Return exactly one label per nugget, in the given order."""

_NUGGET_HUMAN = """Question: {question}

Answer:
{answer}

Nuggets ({count}):
{nuggets}

Return {count} labels."""

_GROUNDED_SYSTEM = """You check an answer from a radiation-safety question-answering \
system against the retrieved context the answer was based on.

Judge only against the context, even if you believe the context is wrong or \
incomplete. Never use your own knowledge.

unsupported_claims: list every factual statement in the answer that the context \
does not support. Paraphrase and translation (Danish/English) are fine. A \
statement is unsupported if its numbers, units (Gy vs Sv, mSv vs µSv), thresholds, \
legal references (paragraph, annex, order number) or jurisdiction (Danish rule \
vs IAEA recommendation) differ from the context, or if it adds facts the context \
does not contain. Do not list statements that only say information is missing, or \
general advice to consult an authority.

refused: true if the answer does not substantively answer the question, e.g. \
states that the sources do not contain the information, even if it adds related \
information; false if it gives an answer."""

_GROUNDED_HUMAN = """Question: {question}

Context:
{context}

Answer:
{answer}"""

_nugget_prompt = ChatPromptTemplate.from_messages(
    [("system", _NUGGET_SYSTEM), ("human", _NUGGET_HUMAN)]
)
_grounded_prompt = ChatPromptTemplate.from_messages(
    [("system", _GROUNDED_SYSTEM), ("human", _GROUNDED_HUMAN)]
)


def judge_item(item: dict, answer: str, context: str, llm) -> dict | None:
    """Judge one answer; returns a verdict for eval.scoring.score_item.

    {"nuggets": [label per nugget], "unsupported_claims": [...], "refused": bool},
    or None if the judge failed twice on either call (scored as judge_error).
    """
    nuggets = item.get("nuggets") or []
    if not answer.strip():
        return {
            "nuggets": ["not_support"] * len(nuggets),
            "unsupported_claims": [],
            "refused": False,
        }

    labels: list[str] = []
    if item["expected_behavior"] == "answer":
        for start in range(0, len(nuggets), MAX_NUGGETS_PER_CALL):
            batch = nuggets[start : start + MAX_NUGGETS_PER_CALL]
            batch_labels = _assign(item["question"], answer, batch, llm)
            if batch_labels is None:
                return None
            labels += batch_labels

    grounded = _with_retry(
        lambda: (_grounded_prompt | llm.with_structured_output(Groundedness)).invoke(
            {"question": item["question"], "context": context, "answer": answer}
        ),
        valid=lambda g: isinstance(g, Groundedness),
        what=f"{item.get('id')}: groundedness",
    )
    if grounded is None:
        return None
    return {
        "nuggets": labels,
        "unsupported_claims": list(grounded.unsupported_claims),
        "refused": bool(grounded.refused),
    }


def _assign(question: str, answer: str, batch: list[dict], llm) -> list[str] | None:
    listing = "\n".join(f"{i}. {n['text']}" for i, n in enumerate(batch, start=1))
    result = _with_retry(
        lambda: (_nugget_prompt | llm.with_structured_output(NuggetLabels)).invoke(
            {
                "question": question,
                "answer": answer,
                "nuggets": listing,
                "count": len(batch),
            }
        ),
        valid=lambda r: isinstance(r, NuggetLabels) and len(r.labels) == len(batch),
        what=f"nugget assignment ({len(batch)} nuggets)",
    )
    return list(result.labels) if result is not None else None


def _with_retry(call, valid, what: str):
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            result = call()
        except Exception as exc:  # judge output is untrusted: any failure is retried
            logger.warning("judge %s failed (attempt %d): %s", what, attempt, exc)
            continue
        if valid(result):
            return result
        logger.warning(
            "judge %s returned an invalid result (attempt %d)", what, attempt
        )
    return None
