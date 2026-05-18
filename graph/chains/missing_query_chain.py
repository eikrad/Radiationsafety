"""Chain to generate a retrieval query for 'missing' information given the question and current context."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from graph.llm_factory import get_llm

system = """You are helping to search a document database for radiation safety (IAEA and Danish legislation). The user asked a question but the current context does not fully answer it. Output a short search query (a few keywords or one short phrase) that would find the MISSING information in such a database.
- Focus on the specific fact or entity that is missing (e.g. "designated facility radioactive waste Denmark", "Danish company authorized disposal").
- Include location or scope if the question asks for it (e.g. Denmark, Danish).
- If a previous attempt hint is provided, use it to focus the query on the specific missing information.
- Output only the search query, no explanation. One short phrase or up to 10 words."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: {question}\n\nContext we already have:\n{context}{reflection_hint}",
        ),
    ]
)


def get_missing_query_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Return a chain that takes question, context, and optional reflection_hint."""
    model = llm or get_llm()
    return prompt | model


def invoke_missing_query_chain(
    question: str,
    context: str,
    llm: BaseChatModel | None = None,
    config: dict | None = None,
    reflection: str = "",
) -> str:
    """Return a short query to retrieve missing information.

    Args:
        reflection: Optional hint from the Reflexion grader describing what was missing
                    in the previous generation attempt. Passed as extra context to the LLM
                    so it can focus the retrieval query on the specific gap.
    """
    chain = get_missing_query_chain(llm)
    cfg = config or {}
    reflection_hint = (
        f"\n\nPrevious attempt failed because: {reflection}\n"
        "Use this to focus the query on the specific missing information."
        if reflection
        else ""
    )
    out = chain.invoke(
        {
            "question": question,
            "context": context or "None.",
            "reflection_hint": reflection_hint,
        },
        config=cfg,
    )
    if hasattr(out, "content"):
        q = out.content
    else:
        q = str(out).strip()
    q = q.strip()[:150] or question
    return q
