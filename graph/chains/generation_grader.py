"""Grade an answer against the context it was written from; name what is missing on failure."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import LenientText, get_llm


class GradeGeneration(BaseModel):
    """Step-by-step verdict; the steps come first so the verdict rests on them."""

    parts: LenientText = Field(
        default="",
        description="Each thing the question asks for, numbered (a value, a condition, "
        "who is responsible, a deadline, ...)",
    )
    assessment: LenientText = Field(
        default="",
        description="For each part: 'answered, supported by <source>', "
        "'answered, not in the Facts', or 'not answered'",
    )
    passed: bool = Field(
        description="True only if every part is answered and supported by the Facts, "
        "with nothing beyond them; false otherwise"
    )
    missing_info: LenientText = Field(
        default="",
        description=(
            "Only when passed is false: one short search phrase (max 15 words) for the "
            "fact, paragraph, annex or table the answer lacks, e.g. 'Bilag 6 "
            "sikringsgrupper nedre værdier Cs-137 BEK 1385' or 'GSR Part 3 "
            "requirement 11 supply chain responsibility'. Empty when passed is true."
        ),
    )


# Modelled on the sufficiency grader (steps, jurisdiction rule, one example):
# the earlier single yes/no on "grounded AND addresses the question" let
# partial answers and answers from the wrong order pass, so the retry path
# never fired (ROADMAP, "The local correction loop never ran").
system = """You are a grader checking an answer against the retrieved context ("Facts") it was written from. Each chunk starts with [Source: ... (Danish law | IAEA | web)].

Work in three steps and reply with one JSON object with the fields parts, assessment, passed and missing_info:
1. parts: list each thing the question asks for. A question with several parts ("which group, how often, and what is required") has several.
2. assessment: for each part, say whether the answer gives it and whether the Facts state it: 'answered, supported by <source>', 'answered, not in the Facts', or 'not answered'.
3. passed: true only if every part is answered AND supported by the Facts. Otherwise false, and missing_info names what to search for next.

Rules:
- A part the answer leaves open, or says it cannot answer from the context, is 'not answered', even when the rest is right.
- For a question about Danish rules, the Danish provision must support the answer; an IAEA value does not, and neither does a rule from a different Danish order than the one that governs the activity.
- Judge against the Facts, not your own knowledge.
- If the Facts contain nothing on the question's topic at all and the answer says so without adding claims, that is a correct refusal: passed is true.

Example question: Under Danish rules, if a sealed source is lost, by when must this be reported, and to which authority?
Example Facts: [Source: Bekendtgørelse om radioaktive stoffer (Danish law)] ... skal straks underrette ... hvis en lukket radioaktiv kilde er bortkommet ...
Example answer: Loss of a sealed source must be reported immediately. The context does not say to which authority.
Example reply: {{"parts": "1. the reporting deadline; 2. the authority to report to", "assessment": "1. answered, supported by Bekendtgørelse om radioaktive stoffer; 2. not answered", "passed": false, "missing_info": "authority notified of a lost sealed source Bekendtgørelse om radioaktive stoffer"}}"""

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
    """Return a grader chain that produces GradeGeneration(..., passed, missing_info)."""
    model = llm or get_llm()
    return prompt | model.with_structured_output(GradeGeneration)
