"""Fakes for reading PDFs in tests: real PDF files, and docling without its models."""

import os

from langchain_core.documents import Document

import ingestion


def write_pdf(path, pages):
    """A real PDF with one line of text per page; "" is a page without a text
    layer, as in a scan."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(out)
    return path


def docling_reading(monkeypatch, results, default=None):
    """docling that gives, per file name, one chunk of text or raises the error;
    default for any other file (an attachment's temporary copy)."""

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return cls()

    class Loader:
        def __init__(self, file_path, **kwargs):
            self.result = results.get(os.path.basename(file_path), default)

        def load(self):
            if isinstance(self.result, Exception):
                raise self.result
            return [Document(page_content=self.result)]

    monkeypatch.setattr(ingestion, "HuggingFaceTokenizer", Tokenizer)
    monkeypatch.setattr(ingestion, "HybridChunker", lambda tokenizer: None)
    monkeypatch.setattr(ingestion, "DoclingLoader", Loader)


def ingest_without_chroma(monkeypatch, tmp_path, *, read_registry=False):
    """ingest() over the PDFs in tmp_path/Bekendtgørelse, and with read_registry
    over the registry sources the test set up; returns what was embedded."""
    added = {}
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(ingestion, "get_embedding_provider", lambda *a: "scaleway")
    monkeypatch.setattr(ingestion, "_clear_chroma_collections", lambda *a, **k: None)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep: object())
    if not read_registry:
        monkeypatch.setattr(
            ingestion, "_load_docs_from_registry", lambda include_iaea=True: ([], [])
        )
    monkeypatch.setattr(
        ingestion,
        "_add_documents_rate_limited",
        lambda docs, name, *a, **k: added.setdefault(name, docs),
    )
    return added
