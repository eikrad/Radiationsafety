"""Ingest IAEA and Danish radiation safety PDFs into two Chroma collections."""

import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

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


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert pdfplumber table (list of rows) to markdown."""
    if not table:
        return ""
    rows = [[str(cell or "") for cell in row] for row in table]
    col_count = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (col_count - len(r)))
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("|" + "|".join(["---"] * col_count) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _load_pdf_with_pdfplumber_tables(file_path: str | Path, source_label: str | None = None) -> list[Document]:
    """Load PDF with pdfplumber. Uses 'lines' strategy for tables with visible grid (e.g. Limits)."""
    import pdfplumber

    label = source_label or str(file_path)
    docs = []
    table_settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    header_row: list[str] | None = None
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            parts = []
            tables = page.extract_tables(table_settings=table_settings)
            for t in tables:
                md = _table_to_markdown(t)
                if md:
                    if header_row is None and t:
                        header_row = [str(c or "") for c in t[0]]
                    elif header_row and t and len(t[0]) == len(header_row) and i > 0:
                        md = _table_to_markdown([header_row] + t)
                    parts.append(md)
            text = page.extract_text() or ""
            if not parts:
                if text:
                    parts.append(text)
            elif text:
                parts.insert(0, text)
            content = "\n\n".join(parts).strip()
            if content:
                doc = Document(
                    page_content=content,
                    metadata={"source": label, "page": i, "total_pages": len(pdf.pages)},
                )
                docs.append(doc)
    return docs


def _load_pdf_with_tables(file_path: str | Path, source_label: str | None = None) -> list[Document]:
    """Load PDF with table extraction. Tries pdfplumber first (best for borderless tables), then PyMuPDF, then PyPDF."""
    label = source_label or str(file_path)
    try:
        docs = _load_pdf_with_pdfplumber_tables(file_path, source_label=label)
        if docs:
            return docs
    except Exception:
        pass
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }
    try:
        loader = PyMuPDFLoader(
            str(file_path),
            extract_tables="markdown",
            mode="page",
            extract_tables_settings=table_settings,
        )
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = label
        return docs
    except Exception:
        pass
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = label
    return docs


def _extract_and_load_attachments(parent_path: Path) -> list[Document]:
    """Extract embedded PDF attachments from a PDF and load them with table support."""
    all_docs = []
    try:
        reader = PdfReader(str(parent_path))
    except Exception:
        return []
    if not hasattr(reader, "attachments") or not reader.attachments:
        return []
    for att_name, content_list in reader.attachments.items():
        for i, content in enumerate(content_list):
            if not isinstance(content, (bytes, bytearray)):
                continue
            suffix = ".pdf" if not str(att_name).lower().endswith(".pdf") else ""
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                continue
            try:
                label = f"{parent_path.name} (Anhang: {att_name})"
                docs = _load_pdf_with_tables(tmp_path, source_label=label)
                for d in docs:
                    d.metadata["document_type"] = "Danish law"
                    d.metadata["parent_document"] = str(parent_path.name)
                all_docs.extend(docs)
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return all_docs


def load_dk_law_docs():
    """Load PDFs from Bekendtgørelse (Danish legislation) directory.

    Uses PyMuPDF with extract_tables='markdown' for proper table extraction,
    and loads embedded PDF attachments (Anhänge) that often contain tables.
    """
    dk_path = DOCS_DIR / "Bekendtgørelse"
    if not dk_path.exists():
        return []
    all_docs = []
    pdf_files = list(dk_path.rglob("*.pdf"))
    for i, pdf_path in enumerate(pdf_files):
        print(f"  [{i + 1}/{len(pdf_files)}] {pdf_path.name}")
        try:
            docs = _load_pdf_with_tables(pdf_path)
            for d in docs:
                d.metadata["document_type"] = "Danish law"
            all_docs.extend(docs)
            attach_docs = _extract_and_load_attachments(pdf_path)
            if attach_docs:
                print(f"    + {len(attach_docs)} pages from attachments")
                all_docs.extend(attach_docs)
        except Exception as e:
            print(f"    Warning: skipped {pdf_path.name}: {e}")
    return all_docs


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
        chunk_size=2500,
        chunk_overlap=200,
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
