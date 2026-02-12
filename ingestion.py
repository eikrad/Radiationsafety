"""Ingest IAEA and Danish radiation safety PDFs into two Chroma collections."""

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


def _get_embeddings():
    """Return embeddings based on LLM_PROVIDER env (gemini or mistral)."""
    import os

    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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


def ingest():
    """Run full ingestion: load PDFs, split, embed, persist to Chroma."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    embeddings = _get_embeddings()

    # IAEA collection
    iaea_docs = load_iaea_docs()
    if iaea_docs:
        iaea_splits = text_splitter.split_documents(iaea_docs)
        Chroma.from_documents(
            documents=iaea_splits,
            collection_name=IAEA_COLLECTION,
            embedding=embeddings,
            persist_directory=str(_CHROMA_DIR),
        )
        print(f"Ingested {len(iaea_splits)} chunks into {IAEA_COLLECTION}")

    # Danish law collection
    dk_docs = load_dk_law_docs()
    if dk_docs:
        dk_splits = text_splitter.split_documents(dk_docs)
        Chroma.from_documents(
            documents=dk_splits,
            collection_name=DK_LAW_COLLECTION,
            embedding=embeddings,
            persist_directory=str(_CHROMA_DIR),
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
