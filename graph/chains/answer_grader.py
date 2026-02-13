"""Chain to grade whether the answer addresses the question."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm


class GradeAnswer(BaseModel):
    """Binary score: answer addresses the question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


system = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)


def get_answer_grader(llm: BaseChatModel | None = None) -> Runnable:
    """Return answer grader for the given LLM. Uses get_llm() if llm is None."""
    model = llm or get_llm()
    return answer_prompt | model.with_structured_output(GradeAnswer)
