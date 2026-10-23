"""Grade an answer against the context it was written from; name what is missing on failure."""

import re
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, BeforeValidator, Field

from graph.llm_factory import LenientText, get_llm

PartStatus = Literal[
    "answered_supported", "answered_not_in_facts", "not_answered", "facts_off_topic"
]


def _as_status(value: object) -> str:
    """Tolerate 'Answered supported', 'answered-supported' and the like; anything
    unrecognised counts as not answered, so a garbled reply never passes."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    allowed = PartStatus.__args__
    return text if text in allowed else "not_answered"


class PartVerdict(BaseModel):
    """One thing the question asks for, and how the answer handles it."""

    part: LenientText = Field(default="", description="What the question asks for")
    quote: LenientText = Field(
        default="",
        description="For answered_supported: the words from the Facts that state "
        "it, copied exactly (shorten with ...). Otherwise empty.",
    )
    status: Annotated[PartStatus, BeforeValidator(_as_status)] = Field(
        description="answered_supported: the answer gives it and the Facts state it; "
        "answered_not_in_facts: the answer gives it but the Facts do not state it; "
        "not_answered: the answer leaves it open or says it cannot answer it; "
        "facts_off_topic: the Facts contain nothing at all on this part's topic"
    )


def _normalise(text: str) -> str:
    text = re.sub(r"[\"'“”„‘’«»]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _quote_in_facts(quote: str, facts: str) -> bool:
    """Every piece of the quote, split at elisions, occurs in the Facts."""
    pieces = [
        piece.strip(" .,;:") for piece in re.split(r"\.\.\.|…", _normalise(quote))
    ]
    pieces = [piece for piece in pieces if piece]
    haystack = _normalise(facts)
    return bool(pieces) and all(piece in haystack for piece in pieces)


class GradeGeneration(BaseModel):
    """Part-by-part assessment; the verdict follows from it, not from the model.

    The model's own overall verdict was unreliable: it listed every part of an
    answer as not answered and still passed it (cs137, ROADMAP "Step G1").
    A supported part also needs a quote the Facts contain: without one the
    grader took the answer's reading of the context for every generator miss.
    """

    parts: list[PartVerdict] = Field(
        default_factory=list,
        description="Each thing the question asks for, with its status",
    )
    missing_info: LenientText = Field(
        default="",
        description=(
            "When any part is not answered_supported: one short search phrase (max 15 "
            "words) for the fact, paragraph, annex or table the answer lacks, e.g. "
            "'Bilag 6 sikringsgrupper nedre værdier BEK 1385' or 'GSR Part 3 "
            "requirement 11 supply chain responsibility'. Otherwise empty."
        ),
    )

    def passes(self, facts: str) -> bool:
        """Every part answered from the Facts, each with a quote they contain, or
        (a correct refusal) the Facts off topic for every part. No parts listed
        is not a pass."""
        statuses = {p.status for p in self.parts}
        if statuses == {"facts_off_topic"}:
            return True
        return statuses == {"answered_supported"} and all(
            _quote_in_facts(p.quote, facts) for p in self.parts
        )


# Modelled on the sufficiency grader (steps, jurisdiction rule, one example).
# The earlier single yes/no on "grounded AND addresses the question" flagged 1
# of 31 failed tune answers, so the retry path never fired.
system = """You are a grader checking an answer against the retrieved context ("Facts") it was written from. Each chunk starts with [Source: ... (Danish law | IAEA | web)].

Reply with one JSON object with the fields parts and missing_info.

parts: list each thing the question asks for, each with a status and a quote.
- A question with several parts ("which group, how often, and what is required") has several.
- A question asking which items or criteria apply has one part per item the Facts give, so an item the answer leaves out shows up as not_answered.
- status is exactly one of:
  answered_supported: the answer gives it and the Facts state it;
  answered_not_in_facts: the answer gives it but the Facts do not state it;
  not_answered: the answer leaves it open, gives it only conditionally, or says it cannot answer it;
  facts_off_topic: the Facts contain nothing at all on this part's topic.
- quote: for answered_supported, copy the words from the Facts that state this part, exactly as written (shorten with ...). Then compare: if the quoted words say something else than the answer (another value, condition, group or order), the status is answered_not_in_facts. Leave quote empty for the other statuses.

missing_info: when any part is not answered_supported, one short search phrase for what is missing; otherwise empty.

Rules:
- For a question about Danish rules, only the Danish provision supports the answer; an IAEA value does not, and neither does a rule from a different Danish order than the one that governs the activity.
- Judge against the Facts, not your own knowledge.
- Use facts_off_topic only when no chunk is on that topic. If chunks are on the topic but lack the specific value or condition, the part is not_answered.

Example question: Under Danish rules, if a sealed source is lost, by when must this be reported, and to which authority?
Example Facts: [Source: Bekendtgørelse om radioaktive stoffer (Danish law)] ... skal straks underrette ... hvis en lukket radioaktiv kilde er bortkommet ...
Example answer: Loss of a sealed source must be reported immediately. The context does not say to which authority.
Example reply: {{"parts": [{{"part": "the reporting deadline", "status": "answered_supported", "quote": "skal straks underrette"}}, {{"part": "the authority to report to", "status": "not_answered", "quote": ""}}], "missing_info": "authority notified of a lost sealed source Bekendtgørelse om radioaktive stoffer"}}

Example question: What is the opening time of the regulator's front desk?
Example Facts: [Source: GSR Part 1 (IAEA)] ... The regulatory body shall be effectively independent ...
Example answer: The context does not contain this information.
Example reply: {{"parts": [{{"part": "the opening time", "status": "facts_off_topic", "quote": ""}}], "missing_info": ""}}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Facts:\n\n{documents}\n\nQuestion: {question}\n\nAnswer: {generation}",
        ),
    ]
)


def get_generation_grader(llm: BaseChatModel | None = None) -> Runnable:
    """Return a grader chain that produces GradeGeneration(parts, missing_info)."""
    model = llm or get_llm()
    return prompt | model.with_structured_output(GradeGeneration)
