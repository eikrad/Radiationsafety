"""Chain to generate an effective web search query from the user question and current context."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from graph.llm_factory import get_llm


system = """You are helping to build a short, effective web search query. Given the user's question and the context we already have (if any), output a single search query that would find the missing or most relevant information on the web.
- Include key terms: topic, location (e.g. Denmark, Danish), and what is being asked (e.g. designated facility, company name, regulation).
- Keep it to one short sentence or a few keywords (under 15 words).
- If the question asks for something country-specific (e.g. Danish, in Denmark), include that in the query.
- Output only the search query, no explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: {question}\n\nContext we already have (from our database):\n{context}",
        ),
    ]
)


def get_search_query_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Return a chain that takes question and context and returns a string (the search query)."""
    model = llm or get_llm()
    return prompt | model


def invoke_search_query_chain(
    question: str,
    context: str,
    llm: BaseChatModel | None = None,
    config: dict | None = None,
) -> str:
    """Return a web search query string from question and context. Strips and truncates."""
    chain = get_search_query_chain(llm)
    cfg = config or {}
    out = chain.invoke({"question": question, "context": context or "None."}, config=cfg)
    if hasattr(out, "content"):
        q = out.content
    else:
        q = str(out).strip()
    return q.strip()[:200] or question
