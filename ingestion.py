"""Ingest IAEA and Danish radiation safety documents into two Chroma collections.

Supports (1) local PDFs in documents/IAEA, documents/IAEA_other, documents/Bekendtgørelse,
(2) Danish legislation from document_sources.yaml via Retsinformation XML (newest version),
(3) IAEA and direct PDFs from document_sources.yaml URLs.
"""

import filecmp
import hashlib
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

from docling.chunking import BaseChunk, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import BaseMetaExtractor
from pypdf import PdfReader
from tqdm import tqdm

from graph.llm_factory import (
    get_embedding_model_name,
    get_embedding_provider,
    get_embeddings,
    query_instruction_template,
)
from ingestion_dk import structure_chunks, xml_to_text

load_dotenv()


class _SimpleMetaExtractor(BaseMetaExtractor):
    """Extracts only primitive metadata; filters out complex nested structures.

    DoclingLoader can return nested dict metadata (e.g., DocMeta) which Chroma
    rejects. This extractor keeps only source, headings, and page info.
    """

    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        meta = {"source": file_path}
        if chunk.meta and chunk.meta.headings:
            meta["headings"] = chunk.meta.headings
        return meta

    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        return {"source": file_path, "num_pages": len(dl_doc.pages)}


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

# Token limit for nomic-embed-text (HybridChunker will respect this)
NOMIC_EMBED_MAX_TOKENS = 512
# Tokenizer model ID that matches embedding model dimensionality/behavior
NOMIC_EMBED_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    """Return (iaea_collection_name, dk_collection_name) for the given embedding provider.

    Scaleway collections carry the model id (SCW_EMBED_MODEL), so several
    Scaleway models can be built side by side and compared.
    """
    if embedding_provider == "scaleway":
        suffix = f"-scw-{get_embedding_model_name('scaleway')}"
        return (f"{IAEA_COLLECTION}{suffix}", f"{DK_LAW_COLLECTION}{suffix}")
    if embedding_provider == "mistral":
        return (f"{IAEA_COLLECTION}-mistral", f"{DK_LAW_COLLECTION}-mistral")
    if embedding_provider == "ollama":
        return (f"{IAEA_COLLECTION}-ollama", f"{DK_LAW_COLLECTION}-ollama")
    return (IAEA_COLLECTION, DK_LAW_COLLECTION)


def _clear_chroma_collections(
    embedding_provider: str | None = None, names: list[str] | None = None
) -> None:
    """Delete the provider's collections (or only `names`) so ingestion recreates them."""
    ep = embedding_provider or get_embedding_provider()
    names = names or list(get_collection_names(ep))
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        for name in names:
            try:
                client.delete_collection(name)
            except Exception:
                pass
    except Exception:
        pass


def download_update_for_source(source_id: str) -> tuple[bool, str]:
    """Download the new version for a source and backup the old one. Returns (success, message)."""

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
                _backup_previous(
                    current_path, path, DOCS_DIR / "backup" / folder, source_id, "pdf"
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
            _backup_previous(
                current_path, path, DOCS_DIR / "backup" / folder, source_id, "pdf"
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


def _backup_previous(
    current_path: Path | None,
    new_path: Path,
    backup_dir: Path,
    source_id: str,
    extension: str,
) -> None:
    """Keep the current file as a dated backup before it is replaced, unless the
    download holds the same document: re-fetching an unchanged version used to
    add a duplicate backup on every ingestion."""
    if not current_path or not current_path.exists():
        return
    if _same_document(current_path, new_path, extension):
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d", time.gmtime(current_path.stat().st_mtime))
    try:
        shutil.copy2(
            str(current_path), str(backup_dir / f"{source_id}_{stamp}.{extension}")
        )
    except OSError:
        pass
    rotate_backups(
        backup_dir, source_id, keep=_MAX_BACKUPS_PER_SOURCE, extension=extension
    )


def _same_document(current_path: Path, new_path: Path, extension: str) -> bool:
    """Byte-identical, or for Retsinformation XML the same law text: a re-export
    can differ in line endings, indentation and element ids on every line."""
    try:
        if filecmp.cmp(current_path, new_path, shallow=False):
            return True
    except OSError:
        return False
    if extension != "xml":
        return False
    text = xml_to_text(current_path)
    return bool(text) and text == xml_to_text(new_path)


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
    # The same law in another layout leaves the file alone, so git shows no change.
    if not (current_file.exists() and _same_document(current_file, xml_path, "xml")):
        _backup_previous(current_file, xml_path, _BACKUP_DIR, source_id, "xml")
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


def _load_docs_from_registry(
    include_iaea: bool = True,
) -> tuple[list[Document], list[Document]]:
    """Fetch from document_sources.yaml: Danish via XML (newest), IAEA/direct via PDF.

    Returns (iaea_docs, dk_docs) — both lists are pre-chunked and ready to embed.
    XML-sourced Danish docs are chunked along their paragraphs (ingestion_dk).
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
                dk_docs.extend(structure_chunks(path, label))
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
        if not include_iaea:
            continue
        path, label = fetch_pdf_for_source(source_id, name, url, folder)
        if path is None:
            continue
        try:
            docs = _load_pdf_with_docling(path, source_label=label)
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
        pdf_files = sorted(base_path.rglob("*.pdf"))
        for pdf_path in tqdm(
            pdf_files, desc="Loading IAEA PDFs", unit="file", disable=False
        ):
            try:
                docs = _load_pdf_with_docling(pdf_path)
                for d in docs:
                    d.metadata["document_type"] = "IAEA"
                all_docs.extend(docs)
            except Exception as e:
                tqdm.write(f"  Warning: skipped {pdf_path.name}: {e}")
    return all_docs


def _load_pdf_with_docling(
    file_path: str | Path,
    source_label: str | None = None,
) -> list[Document]:
    """Load and chunk a PDF using DoclingLoader.

    Returns semantic chunks with source metadata set. Falls back to
    pypdf plain-text extraction if docling fails.
    """
    label = source_label or str(file_path)
    try:
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=NOMIC_EMBED_TOKENIZER_MODEL,
            max_token_count=NOMIC_EMBED_MAX_TOKENS,
        )
        chunker = HybridChunker(tokenizer=tokenizer)
        loader = DoclingLoader(
            file_path=str(file_path),
            chunker=chunker,
            meta_extractor=_SimpleMetaExtractor(),
        )
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = label
        return docs
    except Exception as e:
        tqdm.write(
            f"  Warning: docling failed for {Path(file_path).name}, falling back to pypdf: {e}"
        )
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


def _extract_and_load_attachments(
    parent_path: Path, reader: PdfReader | None = None
) -> list[Document]:
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
                docs = _load_pdf_with_docling(tmp_path, source_label=label)
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


def law_title_key(title: str) -> str:
    """A law's title reduced to lowercase letters and single spaces, so a PDF
    title line ("Bekendtgørelse om strålingsgeneratorer1)") matches the XML
    DocumentTitle."""
    letters = re.sub(
        r"[^a-zæøåäöüé]+", " ", unicodedata.normalize("NFC", title).casefold()
    )
    return re.sub(r"\s+", " ", letters).strip()


def pdf_law_match(first_lines: list[str], law_keys: set[str]) -> str | None:
    """The law key a PDF's title area names, if it is one of law_keys.

    Only the first lines count: a guidance document that cites an order in its
    body is not a copy of that order.
    """
    for line in first_lines:
        key = law_title_key(line)
        for law in law_keys:
            if law and key.startswith(law):
                return law
    return None


def _pdf_first_lines(pdf_path: Path, n: int = 3) -> list[str]:
    """The first n non-empty lines of a PDF's first page."""
    try:
        text = PdfReader(str(pdf_path)).pages[0].extract_text() or ""
    except Exception:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()][:n]


def load_dk_law_docs(skip_law_keys: set[str] | frozenset[str] = frozenset()):
    """Load PDFs from Bekendtgørelse (Danish legislation) directory.

    Uses docling HybridChunker for PDF parsing and loads embedded PDF
    attachments (Anhänge) that often contain tables.

    skip_law_keys: laws (law_title_key) already ingested from Retsinformation
    XML. Their PDFs are skipped: the XML is the current version, and a PDF left
    in the folder after an update would put an outdated version next to it.
    """
    dk_path = DOCS_DIR / "Bekendtgørelse"
    if not dk_path.exists():
        return []
    all_docs = []
    pdf_files = sorted(dk_path.rglob("*.pdf"))
    for pdf_path in tqdm(
        pdf_files, desc="Loading Danish PDFs", unit="file", disable=False
    ):
        if skip_law_keys and pdf_law_match(
            _pdf_first_lines(pdf_path), set(skip_law_keys)
        ):
            tqdm.write(
                f"  Skipped {pdf_path.name}: this law is already ingested from "
                "retsinformation.dk XML (one copy per law)"
            )
            continue
        try:
            docs = _load_pdf_with_docling(pdf_path)
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
                    tqdm.write(f"    + {len(attach_docs)} pages from attachments")
                    all_docs.extend(attach_docs)
        except Exception as e:
            tqdm.write(f"    Warning: skipped {pdf_path.name}: {e}")
    return all_docs


# Scaleway embeddings: chunks per request (1000 short texts were accepted; chunks
# run up to 512 tokens, so stay well below).
SCALEWAY_BATCH_SIZE = 64


def _add_documents_rate_limited(
    documents, collection_name, embeddings, persist_directory, embedding_provider=None
):
    """Add docs. Gemini: batches + optional delay. Ollama and Scaleway: batches with retry."""
    ep = embedding_provider or get_embedding_provider()
    if ep == "gemini":
        _add_documents_gemini_rate_limited(
            documents, collection_name, embeddings, persist_directory
        )
    elif ep == "ollama":
        # small batches and a pause: a local embedding model is easily overloaded
        _add_documents_batched(
            documents, collection_name, embeddings, persist_directory, 10, 0.3
        )
    elif ep == "scaleway":
        _add_documents_batched(
            documents,
            collection_name,
            embeddings,
            persist_directory,
            SCALEWAY_BATCH_SIZE,
        )
    else:
        with tqdm(
            total=1,
            desc=f"Embedding and adding to {collection_name}",
            unit="collection",
            disable=False,
        ) as pbar:
            Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            pbar.update(1)


def _add_documents_batched(
    documents,
    collection_name,
    embeddings,
    persist_directory,
    batch_size: int,
    pause_sec: float = 0.0,
    max_retries: int = 3,
):
    """Add documents in batches, retrying a failed batch (connection errors, rate limits)."""
    vectorstore = None
    num_batches = (len(documents) + batch_size - 1) // batch_size

    with tqdm(
        total=num_batches,
        desc=f"Adding to {collection_name}",
        unit="batch",
        disable=False,
    ) as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            for attempt in range(max_retries):
                try:
                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            collection_name=collection_name,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                    else:
                        vectorstore.add_documents(batch)
                    pbar.update(1)
                    pbar.set_postfix({"chunks": len(batch)})
                    if pause_sec > 0 and i + batch_size < len(documents):
                        time.sleep(pause_sec)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_sec = 1 + (attempt * 2)  # 1s, 3s, 5s
                        tqdm.write(
                            f"  Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {wait_sec}s..."
                        )
                        time.sleep(wait_sec)
                    else:
                        tqdm.write(
                            f"  Batch {batch_num} failed after {max_retries} attempts. Last error: {e}"
                        )
                        raise


def _add_documents_gemini_rate_limited(
    documents, collection_name, embeddings, persist_directory
):
    """Add documents in batches. Delay between batches from GEMINI_BATCH_DELAY_SEC (0 = no delay)."""
    delay_sec = _gemini_batch_delay_sec()
    vectorstore = None
    num_batches = (len(documents) + GEMINI_BATCH_SIZE - 1) // GEMINI_BATCH_SIZE

    with tqdm(
        total=num_batches,
        desc=f"Adding to {collection_name}",
        unit="batch",
        disable=False,
    ) as pbar:
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
            pbar.update(1)
            pbar.set_postfix({"chunks": len(batch)})
            if i + GEMINI_BATCH_SIZE < len(documents) and delay_sec > 0:
                time.sleep(delay_sec)


def ingest(dk_only: bool = False):
    """Run full ingestion: load PDFs (local + from document_sources URLs), embed, persist to Chroma.

    PDF docs are pre-chunked by docling's HybridChunker. XML-sourced Danish docs are chunked
    along their paragraphs inside _load_docs_from_registry; a Danish law read from XML is not read again from a PDF
    copy. dk_only rebuilds only the Danish law collection (the IAEA one stays as it is).
    Embedding provider is determined by EMBEDDING_PROVIDER (LLM_PROVIDER=ollama embeds locally).
    """

    def _signal_handler(signum, frame):
        print("\n\n⚠️  Ingestion interrupted by user. Exiting...\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    print("\n🚀 Starting ingestion pipeline...\n")
    ep = get_embedding_provider()
    iaea_name, dk_name = get_collection_names(ep)
    print(f"📊 Using embedding provider: {ep}")
    print(f"📦 Collections: {dk_name if dk_only else f'{iaea_name}, {dk_name}'}\n")

    with tqdm(
        total=6,
        desc="Overall ingestion progress",
        unit="phase",
        disable=False,
        position=0,
    ) as overall_progress:
        _clear_chroma_collections(ep, names=[dk_name] if dk_only else None)
        embeddings = get_embeddings(ep)
        overall_progress.update(1)

        # Load from document_sources.yaml URLs (Retsinformation XML + IAEA/direct PDFs) — pre-chunked
        iaea_from_url, dk_from_url = _load_docs_from_registry(include_iaea=not dk_only)
        if iaea_from_url:
            tqdm.write(
                f"  ✓ Loaded {len(iaea_from_url)} chunks from registry URLs (IAEA)"
            )
        if dk_from_url:
            tqdm.write(
                f"  ✓ Loaded {len(dk_from_url)} chunks from registry URLs (Danish)"
            )
        overall_progress.update(1)

        # IAEA collection: local dirs + registry URLs — all pre-chunked
        if not dk_only:
            iaea_docs = load_iaea_docs()
            iaea_docs.extend(iaea_from_url)
            if iaea_docs:
                _add_documents_rate_limited(
                    iaea_docs, iaea_name, embeddings, str(_CHROMA_DIR)
                )
                tqdm.write(f"✅ Ingested {len(iaea_docs)} chunks into {iaea_name}")
        overall_progress.update(2)

        # Danish law collection: registry XML + local PDFs not already read from XML
        xml_laws = {
            law_title_key(d.metadata["law_title"])
            for d in dk_from_url
            if d.metadata.get("law_title")
        }
        dk_docs = load_dk_law_docs(skip_law_keys=xml_laws)
        dk_docs.extend(dk_from_url)
        overall_progress.update(1)

        if dk_docs:
            _add_documents_rate_limited(dk_docs, dk_name, embeddings, str(_CHROMA_DIR))
            tqdm.write(f"✅ Ingested {len(dk_docs)} chunks into {dk_name}")
        overall_progress.update(1)

    print("\n🎉 Ingestion complete!")


def reembed_target() -> str:
    """The provider a re-embed writes to: EMBEDDING_PROVIDER, never guessed.

    Deliberately not get_embedding_provider(): with LLM_PROVIDER=ollama that
    returns ollama (privacy mode), and a re-embed meant for another provider
    would replace the local collections.
    """
    from graph.llm_factory import EMBEDDING_PROVIDERS

    target = (os.getenv("EMBEDDING_PROVIDER") or "").strip().lower()
    if target not in EMBEDDING_PROVIDERS:
        raise ValueError(
            "Set EMBEDDING_PROVIDER to the embeddings to build "
            f"({', '.join(EMBEDDING_PROVIDERS)}), e.g. EMBEDDING_PROVIDER=scaleway"
        )
    return target


def reembed_from(source_provider: str, target: str) -> None:
    """Embed the chunks of another provider's collections again with `target`.

    Copies text, metadata and ids unchanged, so embedding models are compared
    on exactly the same chunks (a full ingestion re-parses the PDFs, and a
    different docling version could chunk differently). An earlier copy is
    replaced.
    """
    import chromadb

    sources, targets = (
        get_collection_names(source_provider),
        get_collection_names(target),
    )
    if sources == targets:
        raise ValueError(f"{source_provider} and {target} use the same collections")
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    print(f"\n🔁 Re-embedding {', '.join(sources)} → {', '.join(targets)}\n")
    _clear_chroma_collections(target)
    embeddings = get_embeddings(target)
    for src, dst in zip(sources, targets, strict=True):
        stored = client.get_collection(src).get(include=["documents", "metadatas"])
        if not stored["ids"]:
            raise ValueError(f"Collection {src} is empty; run ingestion first")
        documents = [
            Document(page_content=text, metadata=meta or {}, id=doc_id)
            for doc_id, text, meta in zip(
                stored["ids"], stored["documents"], stored["metadatas"], strict=True
            )
        ]
        _add_documents_rate_limited(
            documents, dst, embeddings, str(_CHROMA_DIR), embedding_provider=target
        )
        tqdm.write(f"✅ Re-embedded {len(documents)} chunks into {dst}")


def _retriever_k() -> int:
    """Chunks returned per collection (IAEA and DK each) for one retrieval query
    (RETRIEVER_K, default 5: chosen by the k test, ROADMAP step 5)."""
    raw = (os.getenv("RETRIEVER_K") or "").strip()
    k = int(raw) if raw else 5
    if k < 1:
        raise ValueError("RETRIEVER_K must be at least 1")
    return k


RETRIEVER_K = _retriever_k()

_retrievers_cache: dict[tuple, tuple] | None = (
    None  # keyed by collections + query instruction
)


def check_embedding_collections_ready(embedding_provider: str) -> tuple[bool, str]:
    """Return (ready, message). If not ready (collections missing or empty), message explains how to build them."""
    if embedding_provider not in ("gemini", "mistral", "ollama", "scaleway"):
        return True, ""
    if embedding_provider == "ollama":
        default_msg = (
            "Local embeddings are not built yet. Run: "
            "LLM_PROVIDER=ollama uv run python ingestion.py"
        )
    elif embedding_provider == "scaleway":
        try:
            model = get_embedding_model_name("scaleway")
        except ValueError as e:
            return False, str(e)
        default_msg = (
            f"Scaleway embeddings ({model}) are not built yet. Run: "
            f"EMBEDDING_PROVIDER=scaleway SCW_EMBED_MODEL={model} "
            "uv run python ingestion.py --reembed-from gemini "
            "(or without --reembed-from for a full ingestion)"
        )
    else:
        default_msg = (
            "Embeddings are not built yet. Set GOOGLE_API_KEY in .env (or export it), "
            "then run: uv run python ingestion.py"
        )
    iaea_name, dk_name = get_collection_names(embedding_provider)
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


def load_chunk_texts(embedding_provider: str) -> dict[str, list[str] | None]:
    """The chunk texts of both collections for this provider, in storage order;
    None for a collection that does not exist."""
    import chromadb

    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    texts: dict[str, list[str] | None] = {}
    for name in get_collection_names(embedding_provider):
        try:
            stored = client.get_collection(name).get(include=["documents"])
        except Exception:
            texts[name] = None
            continue
        texts[name] = [t or "" for t in stored["documents"]]
    return texts


def index_fingerprint(embedding_provider: str) -> dict[str, dict | None]:
    """Per collection, the number of chunks and a hash of their texts.

    Independent of chunk ids and storage order, so the same chunks embedded by
    another model (reembed_from) fingerprint the same, and any re-chunking or
    re-ingestion that changes a chunk shows up.
    """
    fingerprint: dict[str, dict | None] = {}
    for name, texts in load_chunk_texts(embedding_provider).items():
        if texts is None:
            fingerprint[name] = None
            continue
        digests = sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts)
        content = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
        fingerprint[name] = {"chunks": len(texts), "content_hash": content[:12]}
    return fingerprint


def add_single_pdf_to_collection(
    pdf_path: Path, *, folder: str = "IAEA_other", source_label: str | None = None
) -> int:
    """Load one PDF, chunk, embed, and add to the IAEA Chroma collection for current embedding provider. Returns chunk count."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Not a PDF file or file missing")
    label = (source_label or "").strip() or pdf_path.stem.replace("_", " ").replace(
        "-", " "
    )
    splits = _load_pdf_with_docling(pdf_path, source_label=label)
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
    """Return retriever instances for both collections (for use in graph).

    Cached per collection pair and query instruction, so two Scaleway models
    (or the instruction switched off) never share a cached retriever.
    """
    global _retrievers_cache
    if _retrievers_cache is None:
        _retrievers_cache = {}
    ep = (
        embedding_provider
        if embedding_provider in ("gemini", "mistral", "ollama", "scaleway")
        else None
    ) or get_embedding_provider()
    iaea_name, dk_name = get_collection_names(ep)
    template = (
        query_instruction_template(get_embedding_model_name(ep))
        if ep == "scaleway"
        else None
    )
    key = (iaea_name, dk_name, template)
    if key in _retrievers_cache:
        return _retrievers_cache[key]
    embeddings = get_embeddings(ep)
    iaea = Chroma(
        collection_name=iaea_name,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": RETRIEVER_K})
    dk = Chroma(
        collection_name=dk_name,
        embedding_function=embeddings,
        persist_directory=str(_CHROMA_DIR),
    ).as_retriever(search_kwargs={"k": RETRIEVER_K})
    _retrievers_cache[key] = (iaea, dk)
    return _retrievers_cache[key]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=ingest.__doc__.splitlines()[0])
    parser.add_argument(
        "--reembed-from",
        metavar="PROVIDER",
        help="Copy the chunks of this provider's collections (e.g. gemini) and embed "
        "them with the configured EMBEDDING_PROVIDER instead of re-parsing the documents",
    )
    parser.add_argument(
        "--dk-only",
        action="store_true",
        help="Rebuild only the Danish law collection; the IAEA collection is kept",
    )
    args = parser.parse_args()
    if args.reembed_from:
        reembed_from(args.reembed_from, target=reembed_target())
    else:
        ingest(dk_only=args.dk_only)
