"""Ingest IAEA and Danish radiation safety documents into two Chroma collections.

Supports (1) local PDFs in documents/IAEA, documents/IAEA_other, documents/Bekendtgørelse,
(2) Danish legislation from document_sources.yaml via Retsinformation XML (newest version),
(3) IAEA and direct PDFs from document_sources.yaml URLs.
"""

import os
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv
from graph.llm_factory import get_embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "documents"
_BACKUP_DIR = PROJECT_ROOT / "documents" / "backup" / "Bekendtgørelse"
_CHROMA_DIR = PROJECT_ROOT / ".chroma"
_MAX_BACKUPS_PER_SOURCE = 2


def rotate_backups(backup_dir: Path, prefix: str, *, keep: int = 2, extension: str = "xml") -> None:
    """Keep only the `keep` most recent files in backup_dir matching `{prefix}_*.{extension}`; delete older ones."""
    if not backup_dir.exists():
        return
    files = sorted(
        backup_dir.glob(f"{prefix}_*.{extension}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


# Collection names
IAEA_COLLECTION = "radiation-iaea"
DK_LAW_COLLECTION = "radiation-dk-law"

# Google Gemini free tier: 100 embedding requests/min; batch + delay to avoid 429
GEMINI_BATCH_SIZE = 80
GEMINI_BATCH_DELAY_SEC = 65


def _clear_chroma_collections() -> None:
    """Delete both collections so the next from_documents recreates them (full re-ingest)."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        for name in (IAEA_COLLECTION, DK_LAW_COLLECTION):
            try:
                client.delete_collection(name)
            except Exception:
                pass
    except Exception:
        pass


def _xml_to_text(xml_path: Path) -> str:
    """Extract plain text from Retsinformation XML (strip tags, normalize whitespace)."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return ""
    parts: list[str] = []
    for elem in root.iter():
        if elem.text:
            parts.append(elem.text)
        if elem.tail:
            parts.append(elem.tail)
    text = "".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def _load_retsinformation_xml(xml_path: Path, source_label: str) -> list[Document]:
    """Load Retsinformation XML into one or more Document(s). Single doc with full text."""
    text = _xml_to_text(xml_path)
    if not text:
        return []
    doc = Document(
        page_content=text,
        metadata={"source": source_label, "document_type": "Danish law"},
    )
    return [doc]


def download_update_for_source(source_id: str) -> tuple[bool, str]:
    """Download the new version for a source and backup the old one. Returns (success, message)."""
    import time
    try:
        from document_updates import (
            _load_registry,
            _load_versions,
            check_one_source,
            get_local_pdf_path,
            update_registry_url,
            update_version_after_ingest,
        )
        from ingestion_fetch import (
            get_pdf_url_iaea,
            get_xml_url_retsinformation,
            _download_to_temp,
            _download_xml,
        )
    except ImportError as e:
        return False, str(e)
    registry = _load_registry()
    source = next((s for s in registry if (s.id or "").strip() == source_id.strip()), None)
    if not source:
        return False, "Source not found"
    versions = _load_versions()
    result = check_one_source(source, versions)
    if not result.get("update_available") or not result.get("download_url"):
        return False, "No update available for this source"
    download_url = (result.get("download_url") or "").strip()
    remote_label = result.get("remote_version") or source.name
    folder = (source.folder or "IAEA").strip()

    if folder == "Bekendtgørelse":
        if "sst.dk" in download_url.lower():
            # SST vejledninger are PDFs (e.g. vejledning-om-aabne-radioaktive-kilder)
            path = _download_to_temp(download_url)
            if path is None:
                return False, "Failed to download PDF from sst.dk"
            try:
                folder_path = DOCS_DIR / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                current_path = get_local_pdf_path(source)
                backup_dir = DOCS_DIR / "backup" / folder
                if current_path and current_path.exists():
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d", time.gmtime())
                    backup_path = backup_dir / f"{source_id}_{stamp}.pdf"
                    try:
                        shutil.copy2(str(current_path), str(backup_path))
                    except OSError:
                        pass
                    rotate_backups(backup_dir, source_id, keep=_MAX_BACKUPS_PER_SOURCE, extension="pdf")
                dest = current_path if (current_path and current_path.exists()) else None
                if not dest:
                    safe_name = (source.filename_hint or f"{source_id}.pdf").strip()
                    if not safe_name.lower().endswith(".pdf"):
                        safe_name += ".pdf"
                    dest = folder_path / safe_name
                shutil.copy2(str(path), str(dest))
                (folder_path / f"{source_id}_version.txt").write_text(remote_label, encoding="utf-8")
                update_registry_url(source_id, download_url)
                update_version_after_ingest(source_id, remote_label)
                return True, "Downloaded new version and backed up previous."
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        # retsinformation.dk: fetch XML and save as current
        xml_url = get_xml_url_retsinformation(download_url)
        if not xml_url:
            return False, "Could not get XML URL for this document"
        path = _download_xml(xml_url)
        if path is None:
            return False, "Failed to download XML"
        try:
            _save_danish_current_and_trim_backups(source_id, path, version_label=remote_label)
            update_registry_url(source_id, download_url)
            update_version_after_ingest(source_id, remote_label)
            return True, "Downloaded new version and backed up previous."
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    if folder in ("IAEA", "IAEA_other"):
        pdf_url = get_pdf_url_iaea(download_url)
        if not pdf_url:
            return False, "Could not get PDF URL from publication page"
        path = _download_to_temp(pdf_url)
        if path is None:
            return False, "Failed to download PDF"
        try:
            folder_path = DOCS_DIR / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            current_path = get_local_pdf_path(source)
            backup_dir = DOCS_DIR / "backup" / folder
            if current_path and current_path.exists():
                backup_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d", time.gmtime())
                backup_path = backup_dir / f"{source_id}_{stamp}.pdf"
                try:
                    shutil.copy2(str(current_path), str(backup_path))
                except OSError:
                    pass
                rotate_backups(backup_dir, source_id, keep=_MAX_BACKUPS_PER_SOURCE, extension="pdf")
            dest = current_path if (current_path and current_path.exists()) else None
            if not dest:
                safe_name = (source.filename_hint or f"{source_id}.pdf").strip()
                if not safe_name.lower().endswith(".pdf"):
                    safe_name += ".pdf"
                dest = folder_path / safe_name
            try:
                shutil.copy2(str(path), str(dest))
            except OSError as e:
                return False, f"Could not save PDF: {e}"
            update_registry_url(source_id, download_url)
            update_version_after_ingest(source_id, remote_label)
            return True, "Downloaded new version and backed up previous."
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    return False, "Only Bekendtgørelse and IAEA/IAEA_other are supported"


def _save_danish_current_and_trim_backups(
    source_id: str, xml_path: Path, *, version_label: str | None = None
) -> None:
    """Save fetched XML as current for source; move previous current to backup; keep max 2 backups.
    If version_label is set, writes it to {source_id}_version.txt for current-version detection."""
    current_dir = DOCS_DIR / "Bekendtgørelse"
    current_dir.mkdir(parents=True, exist_ok=True)
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    current_file = current_dir / f"{source_id}_current.xml"
    if current_file.exists():
        stamp = time.strftime("%Y%m%d", time.gmtime(current_file.stat().st_mtime))
        backup_path = _BACKUP_DIR / f"{source_id}_{stamp}.xml"
        try:
            current_file.rename(backup_path)
        except OSError:
            pass
    try:
        shutil.copy2(str(xml_path), str(current_file))
    except OSError:
        pass
    if version_label:
        try:
            (current_dir / f"{source_id}_version.txt").write_text(version_label, encoding="utf-8")
        except OSError:
            pass
    rotate_backups(_BACKUP_DIR, source_id, keep=_MAX_BACKUPS_PER_SOURCE)


def _load_docs_from_registry() -> tuple[list[Document], list[Document]]:
    """Fetch from document_sources.yaml: Danish via XML (newest), IAEA/direct via PDF. Returns (iaea_docs, dk_docs)."""
    try:
        from ingestion_fetch import (
            fetch_danish_xml_for_source,
            fetch_pdf_for_source,
            load_sources_registry,
        )
        from document_updates import update_registry_url, update_version_after_ingest
    except ImportError:
        return [], []
    sources = load_sources_registry()
    if not sources:
        return [], []
    iaea_docs: list[Document] = []
    dk_docs: list[Document] = []
    for s in sources:
        source_id = s.get("id") or ""
        name = s.get("name") or "Source"
        url = (s.get("url") or "").strip()
        folder = (s.get("folder") or "IAEA").strip()
        if not url:
            continue
        if folder == "Bekendtgørelse":
            path, label, resolved_url = fetch_danish_xml_for_source(source_id, name, url, use_newest_dk=True)
            if path is None:
                continue
            try:
                docs = _load_retsinformation_xml(path, label)
                for d in docs:
                    d.metadata["document_type"] = "Danish law"
                dk_docs.extend(docs)
                if resolved_url and resolved_url != url:
                    try:
                        update_registry_url(source_id, resolved_url)
                    except Exception:
                        pass
                _save_danish_current_and_trim_backups(source_id, path, version_label=label)
                try:
                    update_version_after_ingest(source_id, label)
                except Exception:
                    pass
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            continue
        # IAEA or other: PDF
        path, label = fetch_pdf_for_source(source_id, name, url, folder)
        if path is None:
            continue
        try:
            docs = PyPDFLoader(str(path)).load()
            for d in docs:
                d.metadata["source"] = label
                d.metadata["document_type"] = "IAEA"
            iaea_docs.extend(docs)
            try:
                update_version_after_ingest(source_id, label)
            except Exception:
                pass
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return iaea_docs, dk_docs


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
    """Run full ingestion: load PDFs (local + from document_sources URLs), split, embed, persist to Chroma."""
    _clear_chroma_collections()

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
    embeddings = get_embeddings()

    # Load from document_sources.yaml URLs (Retsinformation, IAEA, direct PDFs)
    iaea_from_url, dk_from_url = _load_docs_from_registry()
    if iaea_from_url:
        print(f"  Loaded {len(iaea_from_url)} pages from registry URLs (IAEA)")
    if dk_from_url:
        print(f"  Loaded {len(dk_from_url)} pages from registry URLs (Danish)")

    # IAEA collection: local dirs + registry URLs
    iaea_docs = load_iaea_docs()
    iaea_docs.extend(iaea_from_url)
    if iaea_docs:
        iaea_splits = text_splitter_iaea.split_documents(iaea_docs)
        _add_documents_rate_limited(
            iaea_splits,
            IAEA_COLLECTION,
            embeddings,
            str(_CHROMA_DIR),
        )
        print(f"Ingested {len(iaea_splits)} chunks into {IAEA_COLLECTION}")

    # Danish law collection: local dirs + registry URLs
    dk_docs = load_dk_law_docs()
    dk_docs.extend(dk_from_url)
    if dk_docs:
        dk_splits = text_splitter_dk.split_documents(dk_docs)
        _add_documents_rate_limited(
            dk_splits,
            DK_LAW_COLLECTION,
            embeddings,
            str(_CHROMA_DIR),
        )
        print(f"Ingested {len(dk_splits)} chunks into {DK_LAW_COLLECTION}")


_retrievers_cache: tuple | None = None


def add_single_pdf_to_collection(
    pdf_path: Path, *, folder: str = "IAEA_other", source_label: str | None = None
) -> int:
    """Load one PDF, split, embed, and add to the IAEA Chroma collection (does not clear existing). Returns chunk count."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Not a PDF file or file missing")
    label = (source_label or "").strip() or pdf_path.stem.replace("_", " ").replace("-", " ")
    docs = PyPDFLoader(str(pdf_path)).load()
    for d in docs:
        d.metadata["source"] = label
        d.metadata["document_type"] = "IAEA"
    if not docs:
        return 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = text_splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=IAEA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    )
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        for i in range(0, len(splits), GEMINI_BATCH_SIZE):
            batch = splits[i : i + GEMINI_BATCH_SIZE]
            vectorstore.add_documents(batch)
            if i + GEMINI_BATCH_SIZE < len(splits):
                time.sleep(GEMINI_BATCH_DELAY_SEC)
    else:
        vectorstore.add_documents(splits)
    return len(splits)


def clear_retrievers_cache() -> None:
    """Clear the retriever cache so the next query uses fresh Chroma data (e.g. after re-ingestion)."""
    global _retrievers_cache
    _retrievers_cache = None


def get_retrievers():
    """Return retriever instances for both collections (for use in graph). Cached per process."""
    global _retrievers_cache
    if _retrievers_cache is not None:
        return _retrievers_cache
    embeddings = get_embeddings()
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
    _retrievers_cache = (iaea, dk)
    return _retrievers_cache


if __name__ == "__main__":
    ingest()
