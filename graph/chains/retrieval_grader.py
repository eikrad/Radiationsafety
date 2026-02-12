"""Chain to grade document relevance to the question."""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm

llm = get_llm()


class GradeDocuments(BaseModel):
    """Binary score for relevance check."""

    binary_score: bool = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
