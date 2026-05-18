"""Single chain to grade generation quality and extract a reflection hint on failure."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm


class GradeGeneration(BaseModel):
    """Pass/fail verdict and optional reflection hint for the retry loop."""

    passed: bool = Field(
        description=(
            "True if the generation is grounded in the Facts AND fully addresses the question. "
            "False otherwise."
        )
    )
    missing_info: str = Field(
        default="",
        description=(
            "Only when passed=False: one short phrase (max 15 words) naming the specific "
            "fact, section, or document missing from the retrieved context that would fix the answer. "
            "Examples: 'occupational dose limits table Annex 2 GSR-3', "
            "'Danish designated radioactive waste disposal facility name', "
            "'BEK-2025-138405 section 4 notification requirements'. "
            "Leave as empty string when passed=True."
        ),
    )


system = """You are a grader. The "Facts" below are the retrieved context given to the model. \
The "Generation" is the model's answer.

1) passed: Answer YES if the generation's factual content is supported by the Facts \
(paraphrases, summaries, and citing the given sources are fine) AND the generation \
addresses or resolves the user question. Answer NO if there are unsupported claims, \
fabrication, or the question is not answered.

2) missing_info: Only when passed=NO — write one short phrase (max 15 words) naming \
the specific fact, document section, or source missing from the Facts that would fix the answer. \
Examples: 'occupational dose limits table Annex 2 GSR-3', \
'Danish designated radioactive waste disposal facility name'. \
Leave as empty string when passed=YES."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Facts:\n\n{documents}\n\nQuestion: {question}\n\nGeneration: {generation}",
        ),
    ]
)


def get_generation_grader(llm: BaseChatModel | None = None) -> Runnable:
    """Return a grader chain that produces GradeGeneration(passed, missing_info)."""
    model = llm or get_llm()
    return prompt | model.with_structured_output(GradeGeneration)
