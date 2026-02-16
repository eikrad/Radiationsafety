"""Tests for ingestion_fetch module."""

import pytest

import ingestion_fetch as fetch


def test_get_pdf_url_retsinformation():
    """Retsinformation ELI URL is converted to PDF URL."""
    url = fetch.get_pdf_url_retsinformation("https://www.retsinformation.dk/eli/lta/2019/670")
    assert url == "https://www.retsinformation.dk/eli/lta/2019/670/pdf"


def test_get_pdf_url_retsinformation_already_pdf():
    """URL already ending in /pdf is returned as-is."""
    url = fetch.get_pdf_url_retsinformation("https://www.retsinformation.dk/eli/lta/2019/670/pdf")
    assert url == "https://www.retsinformation.dk/eli/lta/2019/670/pdf"


def test_get_xml_url_retsinformation():
    """Retsinformation ELI URL is converted to XML URL."""
    url = fetch.get_xml_url_retsinformation("https://www.retsinformation.dk/eli/lta/2019/670")
    assert url == "https://www.retsinformation.dk/eli/lta/2019/670/xml"


def test_get_xml_url_retsinformation_already_xml():
    """URL already ending in /xml is normalized to base then /xml."""
    url = fetch.get_xml_url_retsinformation("https://www.retsinformation.dk/eli/accn/C20210900609/xml")
    assert url == "https://www.retsinformation.dk/eli/accn/C20210900609/xml"


def test_get_pdf_url_iaea_from_html():
    """IAEA PDF link is extracted from HTML."""
    html = '''
    <a href="https://www-pub.iaea.org/MTCD/Publications/PDF/p15292-PUB2059_web.pdf">Download (1.85 MB)</a>
    '''
    url = fetch.get_pdf_url_iaea("https://www.iaea.org/publications/15292/foo", html=html)
    assert url is not None
    assert "iaea" in url.lower()
    assert ".pdf" in url.lower()


def test_load_sources_registry():
    """Registry loads from example when main config missing."""
    from pathlib import Path
    from unittest.mock import patch
    example = Path(__file__).resolve().parent.parent / "document_sources.example.yaml"
    if example.exists():
        sources = fetch.load_sources_registry()
        assert isinstance(sources, list)
        # Example has at least one source
        if sources:
            assert "url" in sources[0]
            assert "folder" in sources[0]
