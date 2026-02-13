"""RAG generation chain with optional chat history for follow-up questions."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.llm_factory import get_llm

llm = get_llm()

RAG_SYSTEM = """You are an assistant for IAEA and Danish radiation safety documents. Use the following retrieved context to answer the question. If you cannot find the answer in the context, say so. Be concise and cite the source when relevant."""

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
generation_chain = prompt | llm | StrOutputParser()
