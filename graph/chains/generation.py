"""RAG generation chain with optional chat history for follow-up questions."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from graph.llm_factory import get_llm

RAG_SYSTEM = """You answer questions on radiation protection and safety from the retrieved context: IAEA safety standards and Danish orders on ionising radiation. Each context block is labeled with [Source: ...]; sources may be document names or web URLs. Web sources may be used and cited when they are relevant to the question.

Rules:
1. Use only information from the retrieved context, not your own knowledge.
2. Keep the meaning of requirements, conditions, definitions and limits exact: do not change how binding a rule is (shall is not should), how its conditions combine (and is not or), what or whom it applies to, or any number, unit or legal reference. Quote the source when the exact words matter.
3. Say whether a statement comes from a Danish order or from an IAEA standard. An IAEA requirement is not a Danish rule: for a question about Danish rules, answer from the Danish order and label anything you add from IAEA as IAEA.
4. When the context defines the term asked about or lists the items asked for, give the whole definition or list.
5. If the context does not answer the question, or only part of it, say so and name what is missing. Add only what the context states that bears directly on the question; do not fill the gap.
6. The documents cover ionising radiation. Non-ionising radiation (radiofrequency fields, static and low-frequency fields, optical radiation) is outside their scope: say so and do not answer, even if a web source in the context does.
7. Be concise and cite the source of each statement."""

RAG_TEMPLATE = """{context}

{chat_history}

User: {question}
Assistant:"""


def get_generation_chain(
    llm: BaseChatModel | None = None, system: str = RAG_SYSTEM
) -> Runnable:
    """Return RAG generation chain for the given LLM. Uses get_llm() if llm is None.

    system replaces the system prompt (eval.prompt_test compares prompts)."""
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", RAG_TEMPLATE),
        ]
    )
    return prompt | model | StrOutputParser()
