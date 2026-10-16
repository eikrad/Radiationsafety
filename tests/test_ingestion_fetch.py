"""Tests for ingestion_fetch module."""

import ingestion_fetch as fetch


def test_get_pdf_url_retsinformation():
    """Retsinformation ELI URL is converted to PDF URL."""
    url = fetch.get_pdf_url_retsinformation(
        "https://www.retsinformation.dk/eli/lta/2019/670"
    )
    assert url == "https://www.retsinformation.dk/eli/lta/2019/670/pdf"


def test_get_pdf_url_retsinformation_already_pdf():
    """URL already ending in /pdf is returned as-is."""
    url = fetch.get_pdf_url_retsinformation(
        "https://www.retsinformation.dk/eli/lta/2019/670/pdf"
    )
    assert url == "https://www.retsinformation.dk/eli/lta/2019/670/pdf"


def test_get_xml_url_retsinformation():
    """Retsinformation ELI URL is converted to XML URL."""
    url = fetch.get_xml_url_retsinformation(
        "https://www.retsinformation.dk/eli/lta/2019/670"
    )
    assert url == "https://www.retsinformation.dk/eli/lta/2019/670/xml"


def test_get_xml_url_retsinformation_already_xml():
    """URL already ending in /xml is normalized to base then /xml."""
    url = fetch.get_xml_url_retsinformation(
        "https://www.retsinformation.dk/eli/accn/C20210900609/xml"
    )
    assert url == "https://www.retsinformation.dk/eli/accn/C20210900609/xml"


def test_get_pdf_url_iaea_from_html():
    """IAEA PDF link is extracted from HTML."""
    html = """
    <a href="https://www-pub.iaea.org/MTCD/Publications/PDF/p15292-PUB2059_web.pdf">Download (1.85 MB)</a>
    """
    url = fetch.get_pdf_url_iaea(
        "https://www.iaea.org/publications/15292/foo", html=html
    )
    assert url is not None
    assert "iaea" in url.lower()
    assert ".pdf" in url.lower()


def test_load_sources_registry():
    """Registry loads from example when main config missing."""
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "document_sources.example.yaml"
    if example.exists():
        sources = fetch.load_sources_registry()
        assert isinstance(sources, list)
        # Example has at least one source
        if sources:
            assert "url" in sources[0]
            assert "folder" in sources[0]


# --- why a source could not be fetched --------------------------------------------

import io  # noqa: E402
import urllib.error  # noqa: E402

import pytest  # noqa: E402

IAEA_PAGE = "https://www.iaea.org/publications/11113/occupational-radiation-protection"
LAW = "https://www.retsinformation.dk/eli/lta/2025/1385"


class _Response(io.BytesIO):
    """What urlopen returns: a body with its headers, used in a with block."""

    def __init__(self, body: bytes):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


def _serving(monkeypatch, answer):
    """urlopen that answers every request with this body, or raises this error."""

    def urlopen(req, timeout=None, context=None):
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)


def test_an_iaea_page_that_refuses_the_fetcher_names_the_http_error(monkeypatch):
    """Every IAEA publication page answered 403 on 2026-10-16, and the sources
    were left out of the index without a word."""
    _serving(monkeypatch, urllib.error.HTTPError(IAEA_PAGE, 403, "Forbidden", {}, None))

    with pytest.raises(fetch.FetchError) as failed:
        fetch.fetch_pdf_for_source(
            "iaea-gsg-7", "Occupational Radiation Protection", IAEA_PAGE, "IAEA"
        )

    assert "HTTP Error 403: Forbidden" in str(failed.value)
    assert IAEA_PAGE in str(failed.value)


def test_an_iaea_page_without_a_pdf_link_says_so(monkeypatch):
    _serving(monkeypatch, b"<html><body>Order a printed copy</body></html>")

    with pytest.raises(fetch.FetchError, match="no PDF link"):
        fetch.fetch_pdf_for_source(
            "iaea-gsg-7", "Occupational Radiation Protection", IAEA_PAGE, "IAEA"
        )


def test_a_pdf_url_that_serves_a_web_page_is_not_taken_for_the_pdf(monkeypatch):
    url = "https://www.sst.dk/media/gtqpr5bg/vejledning.pdf"
    _serving(monkeypatch, b"<!DOCTYPE html><html>Siden findes ikke</html>")

    with pytest.raises(fetch.FetchError, match="not a PDF"):
        fetch.fetch_pdf_for_source("dk-vejledning", "Vejledning", url, "IAEA_other")


def test_a_law_retsinformation_cannot_deliver_names_the_network_error(monkeypatch):
    _serving(
        monkeypatch,
        urllib.error.URLError("[Errno -3] Temporary failure in name resolution"),
    )
    monkeypatch.setattr(
        fetch, "_resolve_newest_dk_url", lambda source_id, url, name: (url, name)
    )

    with pytest.raises(fetch.FetchError) as failed:
        fetch.fetch_danish_xml_for_source(
            "dk-stoffer", "Bekendtgørelse om radioaktive stoffer", LAW
        )

    assert "Temporary failure in name resolution" in str(failed.value)
    assert f"{LAW}/xml" in str(failed.value)


def test_the_update_check_still_gets_none_for_a_failed_download(monkeypatch):
    """download_update_for_source words its own message from None."""
    _serving(monkeypatch, urllib.error.HTTPError(IAEA_PAGE, 403, "Forbidden", {}, None))

    assert fetch._download_to_temp("https://www-pub.iaea.org/x.pdf") is None
    assert fetch._download_xml(f"{LAW}/xml") is None
    assert fetch.get_pdf_url_iaea(IAEA_PAGE) is None
