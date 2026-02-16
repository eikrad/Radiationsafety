"""Fetch documents from URLs or APIs for ingestion (Retsinformation, IAEA, direct PDF)."""

import re
import ssl
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allowlist for URLs (same as document_updates)
_ALLOWED_HOSTS = frozenset({
    "www.retsinformation.dk",
    "retsinformation.dk",
    "api.retsinformation.dk",
    "www.iaea.org",
    "iaea.org",
    "www-pub.iaea.org",
    "sst.dk",
    "www.sst.dk",
})
_TIMEOUT = 30
_MAX_SIZE = 50 * 1024 * 1024  # 50 MB per PDF
_MAX_XML_SIZE = 5 * 1024 * 1024  # 5 MB per XML document


def _allowed(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        host = (urlparse(url).netloc or "").lower().lstrip("www.")
        if host in _ALLOWED_HOSTS:
            return True
        if host.endswith(".dk") and "retsinformation" in url:
            return True
        return "iaea.org" in host or "retsinformation" in url
    except Exception:
        return False


def _download_to_temp(url: str) -> Path | None:
    """Download URL to a temporary file. Returns path or None on failure."""
    if not _allowed(url):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "RadiationSafetyRAG/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ssl.create_default_context()) as resp:
            size = int(resp.headers.get("Content-Length") or 0)
            if size > _MAX_SIZE:
                return None
            data = resp.read(_MAX_SIZE + 1)
            if len(data) > _MAX_SIZE:
                return None
            if not data[:4].startswith(b"%PDF"):
                return None
            fd, path = tempfile.mkstemp(suffix=".pdf")
            with open(fd, "wb") as f:
                f.write(data)
            return Path(path)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None


def get_pdf_url_retsinformation(eli_url: str) -> str | None:
    """Return the PDF URL for a retsinformation.dk ELI page (e.g. .../eli/lta/2019/670 -> .../eli/lta/2019/670/pdf)."""
    if not _allowed(eli_url):
        return None
    base = eli_url.rstrip("/")
    if base.endswith("/pdf"):
        return base
    if "/eli/lta/" in base:
        return f"{base}/pdf"
    return None


def get_xml_url_retsinformation(eli_url: str) -> str | None:
    """Return the XML URL for a retsinformation.dk ELI page (e.g. .../eli/lta/2019/670 -> .../eli/lta/2019/670/xml)."""
    if not _allowed(eli_url):
        return None
    base = eli_url.rstrip("/")
    for suffix in ("/pdf", "/xml", "/html"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if "/eli/lta/" in base or "/eli/accn/" in base:
        return f"{base}/xml"
    return None


def _download_xml(url: str) -> Path | None:
    """Download URL to a temporary XML file. Returns path or None on failure."""
    if not _allowed(url):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "RadiationSafetyRAG/1.0", "Accept": "application/xml, text/xml, */*"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ssl.create_default_context()) as resp:
            size = int(resp.headers.get("Content-Length") or 0)
            if size > _MAX_XML_SIZE:
                return None
            data = resp.read(_MAX_XML_SIZE + 1)
            if len(data) > _MAX_XML_SIZE:
                return None
            fd, path = tempfile.mkstemp(suffix=".xml")
            with open(fd, "wb") as f:
                f.write(data)
            return Path(path)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None


def get_pdf_url_iaea(publication_page_url: str, html: str | None = None) -> str | None:
    """Extract PDF download URL from IAEA publication page. If html is None, fetches the page."""
    if not _allowed(publication_page_url):
        return None
    if html is None:
        req = urllib.request.Request(
            publication_page_url,
            headers={"User-Agent": "RadiationSafetyRAG/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ssl.create_default_context()) as resp:
                html = resp.read(50000).decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
    # Match href to PDF (full URL or path to MTCD/Publications/PDF)
    m = re.search(
        r'href="(https?://[^"]+\.pdf[^"]*)"',
        html,
        re.IGNORECASE,
    )
    if m:
        url = m.group(1).split('"')[0].split(" ")[0]
        if "iaea" in url.lower():
            return url
    m = re.search(
        r'href="([^"]*MTCD/Publications/PDF/[^"]+\.pdf[^"]*)"',
        html,
        re.IGNORECASE,
    )
    if m:
        path = m.group(1).strip()
        if not path.startswith("http"):
            path = "https://www-pub.iaea.org" + (path if path.startswith("/") else "/" + path)
        return path
    # Some pages use relative or same-origin PDF path
    m = re.search(
        r'href="(/[^"]*\.pdf[^"]*)"',
        html,
        re.IGNORECASE,
    )
    if m:
        path = m.group(1).strip()
        return "https://www.iaea.org" + path
    return None


def _resolve_newest_dk_url(source_id: str, url: str, name: str) -> tuple[str, str]:
    """Resolve newest Danish document URL from registry/check. Returns (resolved_eli_url, label)."""
    try:
        from document_updates import check_one_source, _load_registry, _load_versions
        registry = _load_registry()
        src = next((s for s in registry if s.id == source_id), None)
        if not src:
            return url, name
        r = check_one_source(src, _load_versions())
        resolved = (r.get("download_url") or url).strip()
        label = r.get("remote_version") or name
        return resolved, label
    except Exception:
        return url, name


def _get_current_version_label(source_id: str) -> str | None:
    """Return the stored version label for a source (from document_versions or registry), or None."""
    try:
        from document_updates import _load_versions, _load_registry
        versions = _load_versions()
        v = versions.get(source_id, {}).get("version")
        if v:
            return v
        registry = _load_registry()
        src = next((s for s in registry if s.id == source_id), None)
        if src and getattr(src, "version", None):
            return src.version
    except Exception:
        pass
    return None


def _label_from_danish_xml(xml_path: Path) -> str | None:
    """Try to extract a BEK version label from Danish XML content (first 2k chars)."""
    try:
        text = xml_path.read_text(encoding="utf-8", errors="ignore")[:2048]
        m = re.search(r"BEK\s+nr\s+\d+\s+af\s+\d{1,2}/\d{1,2}/\d{4}", text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    except Exception:
        pass
    return None


def fetch_danish_xml_for_source(
    source_id: str,
    name: str,
    url: str,
    *,
    use_newest_dk: bool = True,
) -> tuple[Path | None, str, str]:
    """
    Fetch Danish XML. Prefer current registry URL; only resolve newest and update URL when current cannot be reached.
    Returns (temp_xml_path, label, resolved_eli_url). resolved_eli_url equals url when we used current URL (no registry update).
    """
    from urllib.parse import urlparse
    host = (urlparse(url).netloc or "").lower()
    if "retsinformation.dk" not in host or "api." in host:
        return None, name, url
    label = name
    # 1) Try current URL first (keep original unless it can no longer be reached)
    # Require substantial content so we don't accept redirect/error pages as valid XML
    _MIN_DANISH_XML_BYTES = 5000
    xml_url = get_xml_url_retsinformation(url)
    if xml_url:
        path = _download_xml(xml_url)
        if path is not None:
            try:
                content = path.read_bytes()
                if len(content) >= _MIN_DANISH_XML_BYTES and (b"<?xml" in content[:300] or b"<" in content[:500]):
                    label = _get_current_version_label(source_id) or _label_from_danish_xml(path) or name
                    return path, label, url
            except OSError:
                pass
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    # 2) Current URL failed or unreachable — resolve newest and fetch from there
    eli_url = url
    if use_newest_dk:
        eli_url, label = _resolve_newest_dk_url(source_id, url, name)
    xml_url = get_xml_url_retsinformation(eli_url)
    if not xml_url:
        return None, name, eli_url
    path = _download_xml(xml_url)
    return path, label, eli_url


def fetch_pdf_for_source(
    source_id: str,
    name: str,
    url: str,
    folder: str,
    *,
    use_newest_dk: bool = True,
) -> tuple[Path | None, str]:
    """
    Resolve and download the PDF for a document source. Returns (temp_path, label) or (None, name).
    Danish (Bekendtgørelse) sources should use fetch_danish_xml_for_source instead.
    If use_newest_dk and url is retsinformation, tries to get newest version first.
    """
    from urllib.parse import urlparse
    host = (urlparse(url).netloc or "").lower()
    label = name

    pdf_url: str | None = None
    if "retsinformation.dk" in host and "api." not in host:
        if use_newest_dk:
            eli_url, label = _resolve_newest_dk_url(source_id, url, name)
            url = eli_url
        pdf_url = get_pdf_url_retsinformation(url)
    elif "iaea.org" in host:
        pdf_url = get_pdf_url_iaea(url)
    elif url.lower().endswith(".pdf") and _allowed(url):
        pdf_url = url

    if not pdf_url:
        return None, name
    path = _download_to_temp(pdf_url)
    return path, label


def load_sources_registry() -> list[dict[str, Any]]:
    """Load document_sources.yaml (or .example). Returns list of source dicts with id, name, url, folder. Delegates to document_updates."""
    try:
        from document_updates import load_registry_raw
        return load_registry_raw()
    except ImportError:
        return []
