"""Chain to grade whether the retrieved context is sufficient to fully answer the question."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.llm_factory import get_llm


class GradeSufficiency(BaseModel):
    """Step-by-step sufficiency verdict; the steps come first so the verdict rests on them."""

    needed: str = Field(
        default="",
        description="What a complete answer needs: the specific facts, and which "
        "jurisdiction's rule applies (Danish law, the IAEA standards, or either)",
    )
    found: str = Field(
        default="",
        description="Where the context states each needed fact (source and a few "
        "words of the passage), or 'not found'",
    )
    binary_score: bool = Field(
        description="The context is sufficient to fully and correctly answer the question, 'yes' or 'no'"
    )


# Steps and one worked example: aspect prompting was the largest measured gain
# for LLM relevance labels (Thomas et al. 2024), and one example raised
# sufficiency-rating accuracy by about 6 points (Joren et al. 2025). The
# jurisdiction rule targets the project's main risk: an IAEA value presented
# where a Danish rule applies (#129).
system = """You are a grader deciding whether the retrieved context is SUFFICIENT to answer the user's question fully and correctly. Each context chunk starts with [Source: ... (Danish law | IAEA | web)].

Work in three steps and reply with one JSON object with the fields needed, found and binary_score:
1. needed: the specific facts a complete answer needs (values, conditions, obligations, the paragraph or annex), and which rule governs: Danish law when the question is about Denmark or Danish rules, the IAEA standards when it names the IAEA, otherwise either.
2. found: for each needed fact, the chunk that states it, or 'not found'.
3. binary_score: true only if every needed fact is stated in the context by a source of the governing jurisdiction, otherwise false.

Rules:
- A value from the IAEA standards does NOT answer a question about Danish rules, even when it is plausible. Danish values, deadlines and conditions can differ, so the Danish rule itself must be in the context.
- Judge the context, not your own knowledge: a fact you know but the context does not state is 'not found'.
- A chunk that is on the topic but does not state the needed value or condition does not count.

Example question: Under Danish rules, how long must records of a worker's doses be kept?
Example context: [Source: GSR Part 3 (IAEA)] ... records of occupational exposure ... shall be retained until the worker attains or would have attained the age of 75 years ...
Example reply: {{"needed": "the Danish retention period for dose records (Danish law governs)", "found": "not found: only the IAEA rule is in the context", "binary_score": false}}"""

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
