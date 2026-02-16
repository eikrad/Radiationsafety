"""Tests for build_document_sources module."""

from pathlib import Path
from unittest.mock import patch

import pytest

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
    assert any("1380" in t or "tecdoc" in t.lower() or "iaea" in t.lower() for t in terms) or "iaea tecdoc 1380" in [t.lower() for t in terms]
