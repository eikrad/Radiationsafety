"""Chain to grade whether the generation is grounded in the documents."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm


class GradeHallucinations(BaseModel):
    """Binary score: generation grounded in facts."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


def get_hallucination_grader(llm: BaseChatModel | None = None) -> Runnable:
    """Return hallucination grader for the given LLM. Uses get_llm() if llm is None."""
    model = llm or get_llm()
    return hallucination_prompt | model.with_structured_output(GradeHallucinations)
