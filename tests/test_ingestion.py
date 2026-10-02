"""Ingestion pipeline tests."""

from types import SimpleNamespace

import ingestion
from ingestion import (
    DK_LAW_COLLECTION,
    IAEA_COLLECTION,
    get_collection_names,
    load_dk_law_docs,
    load_iaea_docs,
)
from ingestion_dk import structure_chunks, xml_to_text


def test_load_iaea_docs_returns_empty_when_no_dirs(tmp_path, monkeypatch):
    """load_iaea_docs returns [] when IAEA dirs do not exist."""
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_iaea_docs()
    assert docs == []


def test_load_iaea_docs_returns_empty_for_empty_iaea_dir(tmp_path, monkeypatch):
    """load_iaea_docs returns [] when IAEA dir exists but has no PDFs."""
    iaea = tmp_path / "IAEA"
    iaea.mkdir()
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_iaea_docs()
    assert docs == []


def test_load_dk_law_docs_returns_empty_when_no_dir(tmp_path, monkeypatch):
    """load_dk_law_docs returns [] when Bekendtgørelse does not exist."""
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_dk_law_docs()
    assert docs == []


def test_load_dk_law_docs_returns_empty_for_empty_dir(tmp_path, monkeypatch):
    """load_dk_law_docs returns [] when Bekendtgørelse exists but has no PDFs."""
    dk = tmp_path / "Bekendtgørelse"
    dk.mkdir()
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_dk_law_docs()
    assert docs == []


def test_rotate_backups_keeps_only_keep_newest(tmp_path):
    """rotate_backups deletes older files so only `keep` most recent remain."""
    import time

    from ingestion import rotate_backups

    (tmp_path / "a_1.xml").write_text("1")
    time.sleep(0.02)
    (tmp_path / "a_2.xml").write_text("2")
    time.sleep(0.02)
    (tmp_path / "a_3.xml").write_text("3")
    rotate_backups(tmp_path, "a", keep=2)
    remaining = sorted(tmp_path.glob("a_*.xml"), key=lambda p: p.stat().st_mtime)
    assert len(remaining) == 2
    assert (tmp_path / "a_1.xml").exists() is False


def test_collection_names():
    """Collection names match expected constants."""
    assert IAEA_COLLECTION == "radiation-iaea"
    assert DK_LAW_COLLECTION == "radiation-dk-law"


def test_get_collection_names_by_provider():
    """get_collection_names returns base names for gemini, -mistral suffix for mistral."""
    iaea_g, dk_g = get_collection_names("gemini")
    assert iaea_g == "radiation-iaea"
    assert dk_g == "radiation-dk-law"
    iaea_m, dk_m = get_collection_names("mistral")
    assert iaea_m == "radiation-iaea-mistral"
    assert dk_m == "radiation-dk-law-mistral"


# --- One collection pair per embedding model ---------------------------------

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


class _FakeEmbeddings(Embeddings):
    """Deterministic vectors of a fixed size, so collections built by different
    'models' are told apart by their dimension."""

    def __init__(self, dim: int, fail_first: int = 0):
        self.dim, self.fail_first, self.calls = dim, fail_first, 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("transient")
        return [[float(len(t))] + [1.0] * (self.dim - 1) for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "_CHROMA_DIR", tmp_path)
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")
    ingestion.clear_retrievers_cache()
    yield tmp_path
    ingestion.clear_retrievers_cache()


def _build_gemini_collections(chroma_dir):
    from langchain_chroma import Chroma

    iaea, dk = get_collection_names("gemini")
    docs = {
        iaea: [
            Document(
                page_content="III.1. For occupational exposure …",
                metadata={"source": "GSR-3.pdf", "document_type": "IAEA"},
            ),
            Document(
                page_content="523. The TI for a package …",
                metadata={"source": "SSR-6.pdf", "document_type": "IAEA"},
            ),
        ],
        dk: [
            Document(
                page_content="§ 14. For erhvervsmæssig bestråling …",
                metadata={"source": "B20250138405.pdf", "document_type": "Danish law"},
            )
        ],
    }
    for name, batch in docs.items():
        Chroma.from_documents(
            batch,
            _FakeEmbeddings(3),
            collection_name=name,
            persist_directory=str(chroma_dir),
        )
    return docs


def _collection(chroma_dir, name):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir)).get_collection(name)


def test_each_scaleway_model_gets_its_own_collections(monkeypatch):
    monkeypatch.setenv("SCW_EMBED_MODEL", "qwen3-embedding-8b")
    qwen = get_collection_names("scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    bge = get_collection_names("scaleway")

    assert qwen == (
        "radiation-iaea-scw-qwen3-embedding-8b",
        "radiation-dk-law-scw-qwen3-embedding-8b",
    )
    assert bge != qwen
    assert get_collection_names("gemini") == ("radiation-iaea", "radiation-dk-law")


def test_reembedding_copies_the_same_chunks_with_new_vectors(chroma_dir, monkeypatch):
    source = _build_gemini_collections(chroma_dir)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: _FakeEmbeddings(4))

    ingestion.reembed_from("gemini", target="scaleway")

    for src, dst in zip(
        get_collection_names("gemini"), get_collection_names("scaleway"), strict=True
    ):
        copied = _collection(chroma_dir, dst).get(
            include=["documents", "metadatas", "embeddings"]
        )
        assert sorted(copied["documents"]) == sorted(
            d.page_content for d in source[src]
        )
        assert sorted(m["source"] for m in copied["metadatas"]) == sorted(
            d.metadata["source"] for d in source[src]
        )
        assert {len(e) for e in copied["embeddings"]} == {4}
    # the source stays untouched
    assert _collection(chroma_dir, get_collection_names("gemini")[0]).count() == 2


def test_reembedding_replaces_an_earlier_copy(chroma_dir, monkeypatch):
    _build_gemini_collections(chroma_dir)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: _FakeEmbeddings(4))

    ingestion.reembed_from("gemini", target="scaleway")
    ingestion.reembed_from("gemini", target="scaleway")

    assert _collection(chroma_dir, get_collection_names("scaleway")[0]).count() == 2


def test_a_transient_embedding_failure_is_retried(chroma_dir, monkeypatch):
    _build_gemini_collections(chroma_dir)
    flaky = _FakeEmbeddings(4, fail_first=1)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: flaky)
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)

    ingestion.reembed_from("gemini", target="scaleway")

    assert _collection(chroma_dir, get_collection_names("scaleway")[0]).count() == 2


def test_missing_scaleway_collections_say_how_to_build_them(chroma_dir):
    ready, message = ingestion.check_embedding_collections_ready("scaleway")

    assert ready is False
    assert "EMBEDDING_PROVIDER=scaleway" in message
    assert "--reembed-from gemini" in message


def test_retrievers_for_different_scaleway_models_are_not_mixed_up(
    chroma_dir, monkeypatch
):
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep=None: _FakeEmbeddings(4))

    qwen_iaea, _ = ingestion.get_retrievers("scaleway")
    monkeypatch.setenv("SCW_EMBED_MODEL", "bge-multilingual-gemma2")
    bge_iaea, _ = ingestion.get_retrievers("scaleway")

    assert qwen_iaea.vectorstore._collection.name.endswith("qwen3-embedding-8b")
    assert bge_iaea.vectorstore._collection.name.endswith("bge-multilingual-gemma2")


@pytest.mark.parametrize("lang", ["en", "de", "da"])
def test_the_user_facing_warning_for_missing_scaleway_embeddings_names_scaleway(lang):
    from graph.i18n import get_warning_embeddings_not_built

    msg = get_warning_embeddings_not_built("scaleway", lang)

    assert "Scaleway" in msg
    assert "GOOGLE_API_KEY" not in msg


def test_reembedding_targets_the_configured_embeddings_not_the_answer_model(
    monkeypatch,
):
    # LLM_PROVIDER=ollama (privacy mode) once redirected a Scaleway re-embed into
    # the local Ollama collections and replaced them
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "scaleway")

    assert ingestion.reembed_target() == "scaleway"


def test_reembedding_without_a_configured_target_refuses_to_guess(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        ingestion.reembed_target()


# --- Danish law from Retsinformation XML -------------------------------------------

RETSINFO_XML = """<?xml version="1.0" encoding="utf-8"?>
<Dokument>
  <Meta>
    <DocumentType>BEK H#LOKDOK04</DocumentType>
    <AccessionNumber>B20250138505</AccessionNumber>
    <DocumentTitle>Bekendtgørelse om radioaktive stoffer</DocumentTitle>
    <DiesSigni>2025-11-18</DiesSigni>
    <Number>1385</Number>
    <Signature>Jonas Egebart</Signature>
  </Meta>
  <Paragraf>
    <Explicatus>§ 7.</Explicatus>
    <Stk>
      <Exitus>
        <Linea>
          <Char>For arealer, der er mindre end eller lig med 1 m</Char>
          <Char formaChar="Superscript">2</Char>
          <Char>, kan aktivitetskoncentrationen bestemmes som middelværdien.</Char>
        </Linea>
        <Linea>
          <Char>Indeksværdien</Char>
          <Char formaChar="Subscript">A</Char>
          <Char>er højst 1 · 10</Char>
          <Char formaChar="Superscript">6</Char>
          <Char>Bq.</Char>
        </Linea>
      </Exitus>
    </Stk>
  </Paragraf>
</Dokument>
"""


def _write_xml(tmp_path, text=RETSINFO_XML):
    path = tmp_path / "dk-radioaktive-stoffer_current.xml"
    path.write_text(text, encoding="utf-8")
    return path


def test_exponents_and_indices_in_danish_law_stay_readable(tmp_path):
    """The PDF text turns 10^6 into "106"; the XML marks superscripts, so keep them."""
    text = xml_to_text(_write_xml(tmp_path))

    assert "1 m^2, kan" in text or "1 m^2 , kan" in text
    assert "1 · 10^6 Bq." in text
    assert "Indeksværdien_A er" in text


def test_the_xml_metadata_block_is_not_indexed_as_law_text(tmp_path):
    text = xml_to_text(_write_xml(tmp_path))

    assert "H#LOKDOK04" not in text
    assert "Jonas Egebart" not in text
    assert text.startswith(
        "Bekendtgørelse om radioaktive stoffer (BEK nr 1385 af 18.11.2025)"
    )


def test_xml_law_chunks_name_their_law_and_version(tmp_path):
    [doc] = structure_chunks(_write_xml(tmp_path), "BEK nr 1385")

    assert doc.metadata["law_title"] == "Bekendtgørelse om radioaktive stoffer"
    assert doc.metadata["doc_id"] == "B20250138505"


# --- one source per Danish law -------------------------------------------------------


def test_a_law_title_key_ignores_case_accents_and_footnote_marks():
    key = ingestion.law_title_key

    assert key("Bekendtgørelse om strålingsgeneratorer1)") == key(
        "Bekendtgørelse om strålingsgeneratorer"
    )
    assert key("BEKENDTGØRELSE OM  Radioaktive stoffer") == key(
        "Bekendtgørelse om radioaktive stoffer"
    )


def test_a_pdf_of_a_law_already_read_from_xml_is_recognised_by_its_title():
    keys = {ingestion.law_title_key("Bekendtgørelse om radioaktive stoffer")}
    match = ingestion.pdf_law_match

    # printed from retsinformation.dk: date and BEK number come first
    assert match(
        [
            "Udskriftsdato: torsdag den 12. februar 2026",
            "BEK nr 1385 af 18/11/2025 (Gældende)",
            "Bekendtgørelse om radioaktive stoffer",
        ],
        keys,
    )
    # an older version has another number but the same title
    assert match(["Bekendtgørelse om radioaktive stoffer1)", "I medfør af"], keys)
    # a guidance document that merely mentions the order further down is kept
    assert not match(
        ["1", "Udarbejdelse af en sikkerhedsvurdering ved brug af åbne", "kilder"],
        keys,
    )


def test_danish_pdfs_of_laws_ingested_from_xml_are_skipped(tmp_path, monkeypatch):
    """One copy per law: the XML from retsinformation.dk is the current version."""
    dk = tmp_path / "Bekendtgørelse"
    dk.mkdir()
    for name in ("B20250138505.pdf", "Brug af aabne radioaktive kilder.pdf"):
        (dk / name).write_bytes(b"%PDF-1.4")
    first_lines = {
        "B20250138505.pdf": [
            "BEK nr 1385 af 18/11/2025",
            "Bekendtgørelse om radioaktive stoffer",
        ],
        "Brug af aabne radioaktive kilder.pdf": [
            "Vejledning",
            "Brug af åbne radioaktive kilder",
        ],
    }
    loaded = []
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(
        ingestion, "_pdf_first_lines", lambda p, n=3: first_lines[p.name]
    )
    monkeypatch.setattr(
        ingestion,
        "_load_pdf_with_docling",
        lambda p, **kw: (
            loaded.append(p.name) or [ingestion.Document(page_content=p.name)]
        ),
    )
    monkeypatch.setattr(ingestion, "_extract_and_load_attachments", lambda *a, **k: [])

    docs = ingestion.load_dk_law_docs(
        skip_law_keys={ingestion.law_title_key("Bekendtgørelse om radioaktive stoffer")}
    )

    assert loaded == ["Brug af aabne radioaktive kilder.pdf"]
    assert [d.page_content for d in docs] == ["Brug af aabne radioaktive kilder.pdf"]


def _fake_ingest(monkeypatch, iaea_from_url, dk_from_url):
    calls = SimpleNamespace(cleared=[], added={}, skip_keys=None, iaea_loaded=False)
    monkeypatch.setattr(ingestion, "get_embedding_provider", lambda *a: "gemini")
    monkeypatch.setattr(ingestion, "get_embeddings", lambda ep: object())
    monkeypatch.setattr(
        ingestion,
        "_clear_chroma_collections",
        lambda ep=None, names=None: calls.cleared.append(names),
    )
    monkeypatch.setattr(
        ingestion,
        "_load_docs_from_registry",
        lambda include_iaea=True: (iaea_from_url if include_iaea else [], dk_from_url),
    )

    def load_iaea():
        calls.iaea_loaded = True
        return []

    def load_dk(skip_law_keys=frozenset()):
        calls.skip_keys = set(skip_law_keys)
        return []

    monkeypatch.setattr(ingestion, "load_iaea_docs", load_iaea)
    monkeypatch.setattr(ingestion, "load_dk_law_docs", load_dk)
    monkeypatch.setattr(
        ingestion,
        "_add_documents_rate_limited",
        lambda docs, name, *a, **k: calls.added.setdefault(name, len(docs)),
    )
    return calls


XML_CHUNK = ingestion.Document(
    page_content="§ 1 …",
    metadata={"law_title": "Bekendtgørelse om radioaktive stoffer", "doc_id": "B1"},
)


def test_ingestion_skips_the_pdfs_of_laws_it_read_from_xml(monkeypatch):
    calls = _fake_ingest(monkeypatch, [], [XML_CHUNK])

    ingestion.ingest()

    assert calls.skip_keys == {
        ingestion.law_title_key("Bekendtgørelse om radioaktive stoffer")
    }


def test_a_danish_only_rebuild_leaves_the_iaea_collection_alone(monkeypatch):
    iaea_name, dk_name = ingestion.get_collection_names("gemini")
    calls = _fake_ingest(
        monkeypatch, [ingestion.Document(page_content="iaea")], [XML_CHUNK]
    )

    ingestion.ingest(dk_only=True)

    assert calls.cleared == [[dk_name]]
    assert calls.iaea_loaded is False
    assert set(calls.added) == {dk_name}


# --- backups only when the version changed ----------------------------------------


def _danish_dirs(tmp_path, monkeypatch):
    docs, backups = tmp_path / "documents", tmp_path / "backup"
    monkeypatch.setattr(ingestion, "DOCS_DIR", docs)
    monkeypatch.setattr(ingestion, "_BACKUP_DIR", backups)
    return docs / "Bekendtgørelse", backups


def test_saving_the_same_danish_version_again_makes_no_backup(tmp_path, monkeypatch):
    current_dir, backups = _danish_dirs(tmp_path, monkeypatch)
    fetched = _write_xml(tmp_path)
    ingestion._save_danish_current_and_trim_backups("dk-stoffer", fetched)

    ingestion._save_danish_current_and_trim_backups("dk-stoffer", fetched)

    assert (current_dir / "dk-stoffer_current.xml").read_text(
        encoding="utf-8"
    ) == RETSINFO_XML
    assert not backups.exists() or list(backups.iterdir()) == []


def test_a_re_export_with_other_layout_but_the_same_law_text_changes_nothing(
    tmp_path, monkeypatch
):
    """Retsinformation's XML can come back with other line endings, indentation
    or element ids: every line differed in git, but the law was the same."""
    current_dir, backups = _danish_dirs(tmp_path, monkeypatch)
    ingestion._save_danish_current_and_trim_backups("dk-stoffer", _write_xml(tmp_path))
    reexport = tmp_path / "reexport.xml"
    reexport.write_bytes(
        RETSINFO_XML.replace("<Paragraf>", '<Paragraf id="id42">')
        .replace("\n", "\r\n    ")
        .encode("utf-8")
    )

    ingestion._save_danish_current_and_trim_backups("dk-stoffer", reexport)

    assert (current_dir / "dk-stoffer_current.xml").read_text(
        encoding="utf-8"
    ) == RETSINFO_XML
    assert not backups.exists() or list(backups.iterdir()) == []


def test_a_new_danish_version_keeps_the_previous_one_as_backup(tmp_path, monkeypatch):
    current_dir, backups = _danish_dirs(tmp_path, monkeypatch)
    ingestion._save_danish_current_and_trim_backups("dk-stoffer", _write_xml(tmp_path))
    newer = tmp_path / "newer.xml"
    newer.write_text(RETSINFO_XML.replace("1385", "1500"), encoding="utf-8")

    ingestion._save_danish_current_and_trim_backups("dk-stoffer", newer)

    assert "1500" in (current_dir / "dk-stoffer_current.xml").read_text(
        encoding="utf-8"
    )
    [backup] = list(backups.iterdir())
    assert "1385" in backup.read_text(encoding="utf-8")


def test_a_pdf_is_backed_up_only_when_the_download_differs(tmp_path):
    current, backups = tmp_path / "GSR-3.pdf", tmp_path / "backup"
    current.write_bytes(b"%PDF v1")
    same, newer = tmp_path / "same.pdf", tmp_path / "newer.pdf"
    same.write_bytes(b"%PDF v1")
    newer.write_bytes(b"%PDF v2")

    ingestion._backup_previous(current, same, backups, "gsr-3", "pdf")
    assert not backups.exists()

    ingestion._backup_previous(current, newer, backups, "gsr-3", "pdf")
    assert [p.read_bytes() for p in backups.iterdir()] == [b"%PDF v1"]
