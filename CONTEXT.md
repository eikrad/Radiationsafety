# Radiation Safety RAG

Answers questions about radiation safety from IAEA standards and Danish law, and says how far the answer is backed by those documents.

## Answering

**Question**:
What a user asks, in English, German or Danish, possibly following up on earlier exchanges.
_Avoid_: Query, prompt

**Answer**:
The reply to one Question together with its sources, any warning, and whether privacy mode or web search was involved.
_Avoid_: Response, generation, result

**Answer trace**:
How an Answer came about: the first retrieval, whether it was judged sufficient, the steps taken and the follow-up searches sent to the search index. Eval scores it; users never see it.
_Avoid_: Debug info, intermediates

**Model choice**:
The answering model a request asks for, with any API keys the user brought along.
_Avoid_: Model, provider settings

**Privacy mode**:
Answering with the local model only: nothing about the Question leaves the machine, so no web search and no tracing. It follows from the model choice and can't be requested separately.
_Avoid_: Local mode, offline mode, Ollama mode

## Sources

**Search index**:
The embedded chunks of the IAEA standards and of Danish law, searched as two collections. Each embedding model has its own index.
_Avoid_: Vector store, database, Chroma

**Trusted documents**:
The chunks an Answer drew from the search index, as opposed to web results; an Answer is verified against them.
_Avoid_: Local sources

**Web search**:
The last resort when the search index can't cover a Question; never in privacy mode, and the operator can switch it off.
_Avoid_: Internet search, Brave

**Database correction loop**:
Searching the search index again for what is missing (named by the context check or by the answer check) before an Answer is given or web search is tried.
_Avoid_: Retry loop, reflexion loop
