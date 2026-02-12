"""RAG generation chain using LangSmith rlm/rag-prompt."""

from langchain_core.output_parsers import StrOutputParser
from langsmith import Client

from graph.llm_factory import get_llm

llm = get_llm()
prompt = Client().pull_prompt("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()
