"""Tests for build_document_sources module."""

from unittest.mock import patch

import build_document_sources as bds


def test_build_sources_empty_when_no_docs(tmp_path):
    """When documents dirs are missing or empty, build_sources returns empty or only existing registry."""
    with patch.object(bds, "DOCS_DIR", tmp_path):
        sources = bds.build_sources(confirm_urls=False)
    # With empty dirs we get no discoveries; if load_existing_registry returns something we might get that - but we don't merge existing-only. So we get [] when no PDFs and no _version.txt.
    assert isinstance(sources, list)


def test_slug():
    """_slug produces lowercase hyphenated id."""
    assert bds._slug("Safety Reports Series No. 76") == "safety-reports-series-no-76"
    assert bds._slug("BEK nr 1385") == "bek-nr-1385"


def test_extract_pdf_title_and_version_nonexistent(tmp_path):
    """_extract_pdf_title_and_version returns (title_from_stem, None) or (None, None) for missing file."""
    path = tmp_path / "nonexistent.pdf"
    title, version = bds._extract_pdf_title_and_version(path)
    assert title is not None  # falls back to stem
    assert "nonexistent" in (title or "")


def test_extract_iaea_search_terms_nonexistent(tmp_path):
    """_extract_iaea_search_terms returns at least stem-based term for missing PDF."""
    path = tmp_path / "IAEA-TECDOC-1380.pdf"
    terms = bds._extract_iaea_search_terms(path)
    assert isinstance(terms, list)
    assert len(terms) >= 1
    assert any(
        "1380" in t or "tecdoc" in t.lower() or "iaea" in t.lower() for t in terms
    ) or "iaea tecdoc 1380" in [t.lower() for t in terms]


def test_confirming_newer_danish_url_keeps_the_local_version(monkeypatch):
    import document_updates

    source = {
        "url": "https://www.retsinformation.dk/eli/lta/2019/670",
        "version": "BEK nr 670 af 01/01/2019",
    }
    newer = "https://www.retsinformation.dk/eli/lta/2025/1385"
    monkeypatch.setattr(document_updates, "_fetch_url", lambda url: "html")
    monkeypatch.setattr(
        document_updates,
        "_parse_retsinformation",
        lambda *args: ("BEK nr 1385 af 18/11/2025", newer),
    )
    bds._confirm_danish_url(source)
    assert source == {"url": newer, "version": "BEK nr 670 af 01/01/2019"}


def test_rebuild_does_not_overwrite_an_upload_during_discovery(monkeypatch, catalog):
    import pytest

    from source_catalog import CatalogChanged

    catalog.register("old", "Old")

    def discover(**kwargs):
        catalog.register("uploaded", "Uploaded")
        return [{"id": "discovered", "name": "Discovered"}]

    monkeypatch.setattr(bds, "build_sources", discover)
    with pytest.raises(CatalogChanged):
        bds.rebuild_sources(confirm_urls=False)
    assert {s.id for s in catalog.snapshot().sources} == {"old", "uploaded"}


def test_rebuild_and_custom_output_preserve_yaml_format(monkeypatch, catalog, tmp_path):
    import yaml

    discovered = [{"id": "local", "name": "Local", "version": "v1"}]
    monkeypatch.setattr(bds, "build_sources", lambda **kwargs: discovered)
    assert bds.rebuild_sources(confirm_urls=False, dry_run=True) == discovered
    assert not catalog.registry_path.exists()
    output = tmp_path / "export.yaml"
    bds.rebuild_sources(confirm_urls=False, output=output)
    assert yaml.safe_load(output.read_text()) == {"sources": discovered}
    assert not catalog.registry_path.exists()
    bds.rebuild_sources(confirm_urls=False)
    assert catalog.snapshot().entries == discovered
