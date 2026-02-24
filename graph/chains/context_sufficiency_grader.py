"""Chain to grade whether the retrieved context is sufficient to fully answer the question."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm


class GradeSufficiency(BaseModel):
    """Binary score: context is sufficient to answer the question."""

    binary_score: bool = Field(
        description="The context is sufficient to fully and correctly answer the question, 'yes' or 'no'"
    )


system = """You are a grader assessing whether a set of retrieved documents is SUFFICIENT to fully and correctly answer the user's question.
Consider: Does the context contain the specific information needed (e.g. names, numbers, locations, procedures)? Would answering only from this context lead to a complete and correct answer?
Give a binary score 'yes' or 'no'. 'Yes' only if the context clearly contains enough information to answer the question fully and correctly. 'No' if the context is vague, incomplete, or missing key facts (e.g. specific companies, facilities, or country-specific details when the question asks for them)."""

sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}\n\nRetrieved context:\n\n{context}"),
    ]
)


def get_context_sufficiency_grader(llm: BaseChatModel | None = None) -> Runnable:
    """Return context sufficiency grader. Uses get_llm() if llm is None."""
    model = llm or get_llm()
    return sufficiency_prompt | model.with_structured_output(GradeSufficiency)
