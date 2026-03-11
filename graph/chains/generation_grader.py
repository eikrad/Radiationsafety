"""Single chain to grade both grounding and answer quality (saves one LLM call)."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm


class GradeGeneration(BaseModel):
    """Scores for grounding and for answering the question."""

    grounded: bool = Field(description="The answer is grounded in the documents, 'yes' or 'no'")
    answers_question: bool = Field(description="The answer addresses the question, 'yes' or 'no'")


system = """You are a grader. The "Facts" below are the retrieved context that was given to the model. The "Generation" is the model's answer.

1) grounded: Answer YES if the generation's factual content is supported by the Facts (paraphrases, summaries, and citing the given sources are fine). Answer NO only for clear unsupported claims or fabrication.
2) answers_question: Answer YES if the generation addresses or resolves the user question.

Reply with two binary scores: grounded (yes/no) and answers_question (yes/no)."""
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
    """Return a single grader for grounded + answers_question. Uses get_llm() if llm is None."""
    model = llm or get_llm()
    return prompt | model.with_structured_output(GradeGeneration)
