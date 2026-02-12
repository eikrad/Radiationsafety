"""Chain to grade whether the answer addresses the question."""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm

llm = get_llm()


class GradeAnswer(BaseModel):
    """Binary score: answer addresses the question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
