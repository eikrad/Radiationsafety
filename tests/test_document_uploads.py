"""Uploads through their module: real files, registry and temporary search index."""

from types import SimpleNamespace

import chromadb
import pytest
import yaml
from langchain_core.documents import Document

import document_updates
import document_uploads
import ingestion
from graph import search_index
from tests.fakes import RecordingEmbeddings
from tests.pdf_fakes import docling_reading, write_pdf

TITLE = "Radiation safety guide for laboratories"
URL = "https://www.iaea.org/publications/123/guide"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = SimpleNamespace(
        documents=tmp_path / "documents",
        registry=tmp_path / "sources.yaml",
        index=tmp_path / "index",
        pdf=write_pdf(tmp_path / "original.pdf", [TITLE]).read_bytes(),
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "scaleway")
    monkeypatch.setattr(ingestion, "_CHROMA_DIR", ws.index)
    monkeypatch.setattr(document_updates, "REGISTRY_PATH", ws.registry)
    monkeypatch.setattr(
        document_uploads, "_lookup_iaea_publication_url_multi", lambda terms: URL
    )
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep: RecordingEmbeddings())
    monkeypatch.setattr(ingestion.time, "sleep", lambda seconds: None)
    docling_reading(monkeypatch, {}, default="A useful paragraph. " * 20)
    return ws


def _upload(ws, filename="guide.pdf", folder="IAEA_other"):
    return document_uploads.add_uploaded_pdf(
        ws.pdf, filename, folder=folder, documents_dir=ws.documents
    )


def _collection(ws):
    name = search_index.collection_names("scaleway")[0]
    return chromadb.PersistentClient(path=str(ws.index)).get_collection(name)


@pytest.mark.parametrize(
    "folder,prefix", [("IAEA", "iaea-"), ("IAEA_other", "iaea-other-")]
)
def test_a_successful_upload_indexes_before_registering_and_reopens_the_index(
    workspace, monkeypatch, folder, prefix
):
    register = document_uploads.append_source_to_registry
    registered = []

    def register_after_indexing(**entry):
        assert _collection(workspace).count() == 1
        register(**entry)
        registered.append(entry)

    reopened = []

    def reopen_after_registration():
        assert workspace.registry.exists()
        reopened.append(True)

    monkeypatch.setattr(
        document_uploads, "append_source_to_registry", register_after_indexing
    )
    monkeypatch.setattr(search_index, "refresh", reopen_after_registration)

    result = _upload(workspace, folder=folder)

    assert result.chunks_added == 1
    assert result.url == URL
    assert result.problems == ()
    assert result.message == f"Added PDF to collection (1 chunks). URL: {URL}"
    assert reopened == [True]
    assert (workspace.documents / folder / "guide.pdf").read_bytes() == workspace.pdf
    stored = yaml.safe_load(workspace.registry.read_text())["sources"]
    assert len(stored) == len(registered) == 1
    assert stored[0] == {
        "id": prefix + "radiation-safety-guide-for-laboratories",
        "name": TITLE,
        "version": TITLE,
        "filename_hint": "guide.pdf",
        "url": URL,
        "folder": folder,
    }


def test_an_unreadable_pdf_stays_on_disk_but_is_not_registered(workspace, monkeypatch):
    monkeypatch.delenv("INGEST_ALLOW_PYPDF_FALLBACK", raising=False)
    docling_reading(monkeypatch, {"guide.pdf": RuntimeError("CUDA out of memory")})

    with pytest.raises(
        ingestion.DoclingFailedError, match="INGEST_ALLOW_PYPDF_FALLBACK=true"
    ):
        _upload(workspace)

    assert not workspace.registry.exists()
    assert (workspace.documents / "IAEA_other" / "guide.pdf").exists()


def test_zero_chunks_is_a_failure_without_registration(workspace, monkeypatch):
    docling_reading(monkeypatch, {"guide.pdf": []})

    with pytest.raises(document_uploads.NoIndexableContent, match="not registered"):
        _upload(workspace)

    assert not workspace.registry.exists()
    assert not workspace.index.exists()
    assert (workspace.documents / "IAEA_other" / "guide.pdf").exists()


@pytest.mark.parametrize("succeed_first_batch", [False, True])
def test_failed_indexing_leaves_registry_unchanged_and_keeps_partial_chunks(
    workspace, monkeypatch, succeed_first_batch
):
    original_registry = "sources:\n  - id: existing\n    name: Existing document\n"
    workspace.registry.write_text(original_registry)
    batch_size = ingestion.SCALEWAY_BATCH_SIZE
    chunks = [
        Document(page_content=f"Chunk {n}: " + "source text " * 20)
        for n in range(batch_size + 1)
    ]
    docling_reading(monkeypatch, {"guide.pdf": chunks})

    class FailingEmbeddings(RecordingEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if not succeed_first_batch or self.calls > 1:
                raise ConnectionError("Embedding unavailable")
            return super().embed_documents(texts)

    embeddings = FailingEmbeddings()
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep: embeddings)

    with pytest.raises(ConnectionError, match="Embedding unavailable"):
        _upload(workspace)

    assert workspace.registry.read_text() == original_registry
    assert _collection(workspace).count() == (batch_size if succeed_first_batch else 0)
    assert embeddings.calls == (4 if succeed_first_batch else 3)
    assert (
        workspace.documents / "IAEA_other" / "guide.pdf"
    ).read_bytes() == workspace.pdf


def test_upload_reports_sparse_text_and_a_missing_publication_url(
    workspace, monkeypatch
):
    workspace.pdf = write_pdf(
        workspace.documents.parent / "scan.pdf", [""] * 12
    ).read_bytes()
    docling_reading(monkeypatch, {"guide.pdf": "IAEA, Vienna"})
    monkeypatch.setattr(
        document_uploads, "_lookup_iaea_publication_url_multi", lambda terms: None
    )

    result = _upload(workspace)

    assert result.chunks_added == 1
    assert result.url is None
    assert "almost no text" in result.message
    assert "No publication URL found" in result.message
    assert workspace.registry.exists()


def test_registration_failure_is_reported_and_keeps_the_indexed_document(
    workspace, monkeypatch
):
    def fail_registration(**entry):
        raise OSError("Registry is read-only")

    monkeypatch.setattr(
        document_uploads, "append_source_to_registry", fail_registration
    )

    with pytest.raises(OSError, match="Registry is read-only"):
        _upload(workspace)

    assert _collection(workspace).count() == 1
    assert not workspace.registry.exists()
    assert (workspace.documents / "IAEA_other" / "guide.pdf").exists()


def test_uploaded_filenames_cannot_escape_the_documents_folder(workspace):
    _upload(workspace, filename="../../guide.pdf")

    stored = yaml.safe_load(workspace.registry.read_text())["sources"][0]
    saved = workspace.documents / "IAEA_other" / stored["filename_hint"]
    assert saved.parent == workspace.documents / "IAEA_other"
    assert saved.read_bytes() == workspace.pdf
