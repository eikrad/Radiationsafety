"""Chain to grade whether the generation is grounded in the documents."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from graph.llm_factory import get_llm


class GradeHallucinations(BaseModel):
    """Binary score: generation grounded in facts."""

    model_config = ConfigDict(title="IsGrounded")
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Judges claims, not wording: one framing sentence, a general remark or a
# "consult the regulator" note used to fail whole answers built only on local
# sources (#136).
system = """You are a grader assessing whether an answer is supported by a set of retrieved facts.

Check only the answer's factual and regulatory claims: values and units, limits, conditions, obligations, deadlines, and which document or paragraph says so. Each of these must be stated in the facts; paraphrase and translation (Danish/English) are fine.

Ignore everything that is not such a claim: introductory or closing sentences, general remarks, disclaimers, advice to consult the authority or a radiation protection expert, and conclusions that follow directly from supported claims.

Give a binary score: 'yes' if every factual and regulatory claim is supported by the facts, 'no' if at least one is not."""

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
