"""Ingest IAEA and Danish radiation safety PDFs into two Chroma collections."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "documents"
_CHROMA_DIR = PROJECT_ROOT / ".chroma"

# Collection names
IAEA_COLLECTION = "radiation-iaea"
DK_LAW_COLLECTION = "radiation-dk-law"

# Google Gemini free tier: 100 embedding requests/min; batch + delay to avoid 429
GEMINI_BATCH_SIZE = 80
GEMINI_BATCH_DELAY_SEC = 65


def _get_embeddings():
    """Return embeddings based on LLM_PROVIDER env (gemini or mistral)."""
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model="mistral-embed")


def load_iaea_docs():
    """Load PDFs from IAEA and IAEA_other directories."""
    iaea_path = DOCS_DIR / "IAEA"
    iaea_other_path = DOCS_DIR / "IAEA_other"
    all_docs = []
    for base_path in [iaea_path, iaea_other_path]:
        if not base_path.exists():
            continue
        loader = DirectoryLoader(
            str(base_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        docs = loader.load()
        for d in docs:
            d.metadata["document_type"] = "IAEA"
        all_docs.extend(docs)
    return all_docs


def load_dk_law_docs():
    """Load PDFs from Bekendtgørelse (Danish legislation) directory."""
    dk_path = DOCS_DIR / "Bekendtgørelse"
    if not dk_path.exists():
        return []
    loader = DirectoryLoader(
        str(dk_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    for d in docs:
        d.metadata["document_type"] = "Danish law"
    return docs


def _add_documents_rate_limited(
    documents, collection_name, embeddings, persist_directory
):
    """Add docs. Gemini free tier: batches + delay. Mistral: all at once."""
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        _add_documents_gemini_rate_limited(
            documents, collection_name, embeddings, persist_directory
        )
    else:
        Chroma.from_documents(
            documents=documents,
            collection_name=collection_name,
            embedding=embeddings,
            persist_directory=persist_directory,
        )


def _add_documents_gemini_rate_limited(
    documents, collection_name, embeddings, persist_directory
):
    """Gemini free tier: 100 req/min → batch + 65s delay between batches."""
    vectorstore = None
    for i in range(0, len(documents), GEMINI_BATCH_SIZE):
        batch = documents[i : i + GEMINI_BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                collection_name=collection_name,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  Added batch {i // GEMINI_BATCH_SIZE + 1} ({len(batch)} chunks)")
        if i + GEMINI_BATCH_SIZE < len(documents):
            print(f"  Waiting {GEMINI_BATCH_DELAY_SEC}s for rate limit...")
            time.sleep(GEMINI_BATCH_DELAY_SEC)


def ingest():
    """Run full ingestion: load PDFs, split, embed, persist to Chroma."""
    text_splitter_iaea = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_splitter_dk = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        separators=["\n\n", "§ ", "\n", ". ", " ", ""],
    )
    embeddings = _get_embeddings()

    # IAEA collection
    iaea_docs = load_iaea_docs()
    if iaea_docs:
        iaea_splits = text_splitter_iaea.split_documents(iaea_docs)
        _add_documents_rate_limited(
            iaea_splits,
            IAEA_COLLECTION,
            embeddings,
            str(_CHROMA_DIR),
        )
        print(f"Ingested {len(iaea_splits)} chunks into {IAEA_COLLECTION}")

    # Danish law collection
    dk_docs = load_dk_law_docs()
    if dk_docs:
        dk_splits = text_splitter_dk.split_documents(dk_docs)
        _add_documents_rate_limited(
            dk_splits,
            DK_LAW_COLLECTION,
            embeddings,
            str(_CHROMA_DIR),
        )
        print(f"Ingested {len(dk_splits)} chunks into {DK_LAW_COLLECTION}")


def get_retrievers():
    """Return retriever instances for both collections (for use in graph)."""
    embeddings = _get_embeddings()
    iaea = Chroma(
        collection_name=IAEA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": 5})
    dk = Chroma(
        collection_name=DK_LAW_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": 5})
    return iaea, dk


if __name__ == "__main__":
    ingest()
