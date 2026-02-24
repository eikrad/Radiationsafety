"""RAG generation chain with optional chat history for follow-up questions."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from graph.llm_factory import get_llm

RAG_SYSTEM = """You are an assistant for IAEA and Danish radiation safety documents. Use the following retrieved context to answer the question. Each context block is labeled with [Source: ...]; sources may be document names or web URLs. Use and cite web sources (URLs) when they are relevant to the question. If you cannot find the answer in the context, say so. Be concise and cite the source when relevant."""

RAG_TEMPLATE = """{context}

{chat_history}

User: {question}
Assistant:"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        ("human", RAG_TEMPLATE),
    ]
)


def get_generation_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Return RAG generation chain for the given LLM. Uses get_llm() if llm is None."""
    model = llm or get_llm()
    return prompt | model | StrOutputParser()
