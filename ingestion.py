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
from langchain_chroma import Chroma
from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from graph.llm_factory import get_embedding_provider, get_embeddings

load_dotenv()

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "documents"
_BACKUP_DIR = PROJECT_ROOT / "documents" / "backup" / "Bekendtgørelse"
_CHROMA_DIR = PROJECT_ROOT / ".chroma"
_MAX_BACKUPS_PER_SOURCE = 2


def rotate_backups(
    backup_dir: Path, prefix: str, *, keep: int = 2, extension: str = "xml"
) -> None:
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


# Base collection names (Gemini/OpenAI share these; Mistral uses -mistral suffix)
IAEA_COLLECTION = "radiation-iaea"
DK_LAW_COLLECTION = "radiation-dk-law"

# Google Gemini: batch size for embeddings. Delay between batches is from GEMINI_BATCH_DELAY_SEC env (0 or unset = no delay, e.g. 65 for free tier).
GEMINI_BATCH_SIZE = 200

# Docling HybridChunker token budgets (approximates previous character-based chunk sizes).
IAEA_MAX_TOKENS = 256  # ~800 chars
DK_MAX_TOKENS = 512  # ~2500 chars

# Cached HybridChunker instances keyed by max_tokens — avoids reloading the tokeniser per file.
_chunker_cache: dict[int, HybridChunker] = {}


def _gemini_batch_delay_sec() -> float:
    """Seconds to wait between Gemini embedding batches. 0 or unset = no delay (paid tier). Set to 65 for free tier."""
    raw = (os.getenv("GEMINI_BATCH_DELAY_SEC") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def get_collection_names(embedding_provider: str) -> tuple[str, str]:
    """Return (iaea_collection_name, dk_collection_name) for the given embedding provider."""
    if embedding_provider == "mistral":
        return (f"{IAEA_COLLECTION}-mistral", f"{DK_LAW_COLLECTION}-mistral")
    return (IAEA_COLLECTION, DK_LAW_COLLECTION)


def _clear_chroma_collections(embedding_provider: str | None = None) -> None:
    """Delete the two collections for the given embedding provider so the next from_documents recreates them."""
    ep = embedding_provider or get_embedding_provider()
    iaea_name, dk_name = get_collection_names(ep)
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        for name in (iaea_name, dk_name):
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
            _download_to_temp,
            _download_xml,
            get_pdf_url_iaea,
            get_xml_url_retsinformation,
        )
    except ImportError as e:
        return False, str(e)
    registry = _load_registry()
    source = next(
        (s for s in registry if (s.id or "").strip() == source_id.strip()), None
    )
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
                    rotate_backups(
                        backup_dir,
                        source_id,
                        keep=_MAX_BACKUPS_PER_SOURCE,
                        extension="pdf",
                    )
                dest = (
                    current_path if (current_path and current_path.exists()) else None
                )
                if not dest:
                    safe_name = (source.filename_hint or f"{source_id}.pdf").strip()
                    if not safe_name.lower().endswith(".pdf"):
                        safe_name += ".pdf"
                    dest = folder_path / safe_name
                shutil.copy2(str(path), str(dest))
                (folder_path / f"{source_id}_version.txt").write_text(
                    remote_label, encoding="utf-8"
                )
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
            _save_danish_current_and_trim_backups(
                source_id, path, version_label=remote_label
            )
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
                rotate_backups(
                    backup_dir, source_id, keep=_MAX_BACKUPS_PER_SOURCE, extension="pdf"
                )
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
    If version_label is set, writes it to {source_id}_version.txt for current-version detection.
    """
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
            (current_dir / f"{source_id}_version.txt").write_text(
                version_label, encoding="utf-8"
            )
        except OSError:
            pass
    rotate_backups(_BACKUP_DIR, source_id, keep=_MAX_BACKUPS_PER_SOURCE)


def _load_docs_from_registry() -> tuple[list[Document], list[Document]]:
    """Fetch from document_sources.yaml: Danish via XML (newest), IAEA/direct via PDF.

    Returns (iaea_docs, dk_docs) — both lists are pre-chunked and ready to embed.
    XML-sourced Danish docs are split here with RecursiveCharacterTextSplitter.
    """
    try:
        from document_updates import update_registry_url, update_version_after_ingest
        from ingestion_fetch import (
            fetch_danish_xml_for_source,
            fetch_pdf_for_source,
            load_sources_registry,
        )
    except ImportError:
        return [], []
    sources = load_sources_registry()
    if not sources:
        return [], []
    text_splitter_dk = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "§ ", "\n", ". ", " ", ""],
    )
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
            path, label, resolved_url = fetch_danish_xml_for_source(
                source_id, name, url, use_newest_dk=True
            )
            if path is None:
                continue
            try:
                docs = _load_retsinformation_xml(path, label)
                for d in docs:
                    d.metadata["document_type"] = "Danish law"
                dk_docs.extend(text_splitter_dk.split_documents(docs))
                if resolved_url and resolved_url != url:
                    try:
                        update_registry_url(source_id, resolved_url)
                    except Exception:
                        pass
                _save_danish_current_and_trim_backups(
                    source_id, path, version_label=label
                )
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
        # IAEA or other: PDF — docling returns pre-chunked docs
        path, label = fetch_pdf_for_source(source_id, name, url, folder)
        if path is None:
            continue
        try:
            docs = _load_pdf_with_docling(path, source_label=label, max_tokens=IAEA_MAX_TOKENS)
            for d in docs:
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


def load_iaea_docs() -> list[Document]:
    """Load PDFs from IAEA and IAEA_other directories."""
    all_docs = []
    for base_path in [DOCS_DIR / "IAEA", DOCS_DIR / "IAEA_other"]:
        if not base_path.exists():
            continue
        for pdf_path in sorted(base_path.rglob("*.pdf")):
            print(f"  {pdf_path.name}")
            try:
                docs = _load_pdf_with_docling(pdf_path, max_tokens=IAEA_MAX_TOKENS)
                for d in docs:
                    d.metadata["document_type"] = "IAEA"
                all_docs.extend(docs)
            except Exception as e:
                print(f"    Warning: skipped {pdf_path.name}: {e}")
    return all_docs


def _load_pdf_with_docling(
    file_path: str | Path,
    source_label: str | None = None,
    max_tokens: int = IAEA_MAX_TOKENS,
) -> list[Document]:
    """Load and chunk a PDF using docling's HybridChunker.

    Returns pre-chunked Documents with source metadata set. Falls back to
    pypdf plain-text extraction if docling fails.
    """
    label = source_label or str(file_path)
    try:
        if max_tokens not in _chunker_cache:
            _chunker_cache[max_tokens] = HybridChunker(max_tokens=max_tokens)
        loader = DoclingLoader(
            file_path=str(file_path),
            chunker=_chunker_cache[max_tokens],
        )
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = label
        return docs
    except Exception as e:
        print(f"  Warning: docling failed for {Path(file_path).name}, falling back to pypdf: {e}")
    # Fallback: pypdf plain-text extraction
    reader = PdfReader(str(file_path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": label, "page": i},
                )
            )
    return docs


def _extract_and_load_attachments(parent_path: Path, reader: PdfReader | None = None) -> list[Document]:
    """Extract embedded PDF attachments from a PDF and load them."""
    all_docs = []
    if reader is None:
        try:
            reader = PdfReader(str(parent_path))
        except Exception:
            return []
    if not hasattr(reader, "attachments") or not reader.attachments:
        return []
    for att_name, content_list in reader.attachments.items():
        for _i, content in enumerate(content_list):
            if not isinstance(content, (bytes, bytearray)):
                continue
            suffix = ".pdf" if not str(att_name).lower().endswith(".pdf") else ""
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    tmp_path = f.name
                    f.write(content)
            except Exception:
                continue
            try:
                label = f"{parent_path.name} (Anhang: {att_name})"
                docs = _load_pdf_with_docling(tmp_path, source_label=label, max_tokens=DK_MAX_TOKENS)
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

    Uses docling HybridChunker for PDF parsing and loads embedded PDF
    attachments (Anhänge) that often contain tables.
    """
    dk_path = DOCS_DIR / "Bekendtgørelse"
    if not dk_path.exists():
        return []
    all_docs = []
    pdf_files = list(dk_path.rglob("*.pdf"))
    for i, pdf_path in enumerate(pdf_files):
        print(f"  [{i + 1}/{len(pdf_files)}] {pdf_path.name}")
        try:
            docs = _load_pdf_with_docling(pdf_path, max_tokens=DK_MAX_TOKENS)
            for d in docs:
                d.metadata["document_type"] = "Danish law"
            all_docs.extend(docs)
            try:
                reader = PdfReader(str(pdf_path))
            except Exception:
                reader = None
            if reader is not None:
                attach_docs = _extract_and_load_attachments(pdf_path, reader=reader)
                if attach_docs:
                    print(f"    + {len(attach_docs)} pages from attachments")
                    all_docs.extend(attach_docs)
        except Exception as e:
            print(f"    Warning: skipped {pdf_path.name}: {e}")
    return all_docs


def _add_documents_rate_limited(
    documents, collection_name, embeddings, persist_directory
):
    """Add docs. Gemini embeddings: batches + optional delay. Mistral: all at once."""
    ep = get_embedding_provider()
    if ep == "gemini":
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
    """Add documents in batches. Delay between batches from GEMINI_BATCH_DELAY_SEC (0 = no delay)."""
    delay_sec = _gemini_batch_delay_sec()
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
        if i + GEMINI_BATCH_SIZE < len(documents) and delay_sec > 0:
            print(f"  Waiting {delay_sec}s for rate limit...")
            time.sleep(delay_sec)


def ingest():
    """Run full ingestion: load PDFs (local + from document_sources URLs), embed, persist to Chroma.

    PDF docs are pre-chunked by docling's HybridChunker. XML-sourced Danish docs are split
    inside _load_docs_from_registry. Embedding provider is determined by LLM_PROVIDER env
    (gemini requires GOOGLE_API_KEY; mistral uses a separate collection suffix).
    """
    ep = get_embedding_provider()
    iaea_name, dk_name = get_collection_names(ep)
    _clear_chroma_collections(ep)
    embeddings = get_embeddings(ep)

    # Load from document_sources.yaml URLs (Retsinformation XML + IAEA/direct PDFs) — pre-chunked
    iaea_from_url, dk_from_url = _load_docs_from_registry()
    if iaea_from_url:
        print(f"  Loaded {len(iaea_from_url)} chunks from registry URLs (IAEA)")
    if dk_from_url:
        print(f"  Loaded {len(dk_from_url)} chunks from registry URLs (Danish)")

    # IAEA collection: local dirs + registry URLs — all pre-chunked
    iaea_docs = load_iaea_docs()
    iaea_docs.extend(iaea_from_url)
    if iaea_docs:
        _add_documents_rate_limited(iaea_docs, iaea_name, embeddings, str(_CHROMA_DIR))
        print(f"Ingested {len(iaea_docs)} chunks into {iaea_name}")

    # Danish law collection: local dirs + registry URLs — all pre-chunked
    dk_docs = load_dk_law_docs()
    dk_docs.extend(dk_from_url)
    if dk_docs:
        _add_documents_rate_limited(dk_docs, dk_name, embeddings, str(_CHROMA_DIR))
        print(f"Ingested {len(dk_docs)} chunks into {dk_name}")


_retrievers_cache: dict[str, tuple] | None = None  # keyed by embedding_provider


def check_embedding_collections_ready(embedding_provider: str) -> tuple[bool, str]:
    """Return (ready, message). If not ready (collections missing or empty), message explains how to build them.

    Applies to both gemini (Gemini/OpenAI) and mistral embedding providers.
    """
    if embedding_provider not in ("gemini", "mistral"):
        return True, ""
    iaea_name, dk_name = get_collection_names(embedding_provider)
    default_msg = (
        "Embeddings are not built yet. Set GOOGLE_API_KEY in .env (or export it), "
        "then run: uv run python ingestion.py"
    )
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        for name in (iaea_name, dk_name):
            try:
                coll = client.get_collection(name)
                if coll.count() == 0:
                    return False, default_msg
            except Exception:
                return False, default_msg
        return True, ""
    except Exception:
        return False, default_msg


def add_single_pdf_to_collection(
    pdf_path: Path, *, folder: str = "IAEA_other", source_label: str | None = None
) -> int:
    """Load one PDF, chunk, embed, and add to the IAEA Chroma collection for current embedding provider. Returns chunk count."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Not a PDF file or file missing")
    label = (source_label or "").strip() or pdf_path.stem.replace("_", " ").replace(
        "-", " "
    )
    splits = _load_pdf_with_docling(pdf_path, source_label=label, max_tokens=IAEA_MAX_TOKENS)
    for d in splits:
        d.metadata["document_type"] = "IAEA"
    if not splits:
        return 0
    ep = get_embedding_provider()
    iaea_name, _ = get_collection_names(ep)
    embeddings = get_embeddings(ep)
    vectorstore = Chroma(
        collection_name=iaea_name,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    )
    if ep == "gemini":
        delay_sec = _gemini_batch_delay_sec()
        for i in range(0, len(splits), GEMINI_BATCH_SIZE):
            batch = splits[i : i + GEMINI_BATCH_SIZE]
            vectorstore.add_documents(batch)
            if i + GEMINI_BATCH_SIZE < len(splits) and delay_sec > 0:
                time.sleep(delay_sec)
    else:
        vectorstore.add_documents(splits)
    return len(splits)


def clear_retrievers_cache() -> None:
    """Clear the retriever cache so the next query uses fresh Chroma data (e.g. after re-ingestion)."""
    global _retrievers_cache
    _retrievers_cache = None


def get_retrievers(embedding_provider: str | None = None):
    """Return retriever instances for both collections (for use in graph). Cached per embedding_provider.

    When embedding_provider is None, uses get_embedding_provider() (currently always 'gemini').
    Retrieval always uses Gemini embeddings; the same collections are used regardless of LLM for generation.
    """
    global _retrievers_cache
    if _retrievers_cache is None:
        _retrievers_cache = {}
    ep = (
        embedding_provider if embedding_provider in ("gemini", "mistral") else None
    ) or get_embedding_provider()
    if ep in _retrievers_cache:
        return _retrievers_cache[ep]
    iaea_name, dk_name = get_collection_names(ep)
    embeddings = get_embeddings(ep)
    iaea = Chroma(
        collection_name=iaea_name,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": 3})
    dk = Chroma(
        collection_name=dk_name,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": 3})
    _retrievers_cache[ep] = (iaea, dk)
    return _retrievers_cache[ep]


if __name__ == "__main__":
    ingest()
