"""Chain to generate a retrieval query for 'missing' information given the question and current context."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from graph.llm_factory import get_llm


system = """You are helping to search a document database for radiation safety (IAEA and Danish legislation). The user asked a question but the current context does not fully answer it. Output a short search query (a few keywords or one short phrase) that would find the MISSING information in such a database.
- Focus on the specific fact or entity that is missing (e.g. "designated facility radioactive waste Denmark", "Danish company authorized disposal").
- Include location or scope if the question asks for it (e.g. Denmark, Danish).
- Output only the search query, no explanation. One short phrase or up to 10 words."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: {question}\n\nContext we already have:\n{context}",
        ),
    ]
)


def get_missing_query_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Return a chain that takes question and context and returns a string (query for retrieval)."""
    model = llm or get_llm()
    return prompt | model


def invoke_missing_query_chain(
    question: str,
    context: str,
    llm: BaseChatModel | None = None,
    config: dict | None = None,
) -> str:
    """Return a short query to retrieve missing information. Falls back to question if empty."""
    chain = get_missing_query_chain(llm)
    cfg = config or {}
    out = chain.invoke({"question": question, "context": context or "None."}, config=cfg)
    if hasattr(out, "content"):
        q = out.content
    else:
        q = str(out).strip()
    q = q.strip()[:150] or question
    return q
