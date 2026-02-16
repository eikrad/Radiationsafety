"""Check for updated versions of registered document sources (IAEA, retsinformation.dk)."""

import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

# Brave Search API: min seconds between requests to avoid 429 Too Many Requests
BRAVE_REQUEST_DELAY_SECONDS = 4
_brave_throttle_lock = threading.Lock()
_brave_last_request_time: list[float] = [0.0]


def _brave_throttle() -> None:
    """Wait so at least BRAVE_REQUEST_DELAY_SECONDS have passed since the last Brave API call."""
    with _brave_throttle_lock:
        now = time.monotonic()
        elapsed = now - _brave_last_request_time[0]
        if elapsed < BRAVE_REQUEST_DELAY_SECONDS:
            time.sleep(BRAVE_REQUEST_DELAY_SECONDS - elapsed)
        _brave_last_request_time[0] = time.monotonic()

# Paths (aligned with ingestion)
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "documents"
REGISTRY_PATH = PROJECT_ROOT / "document_sources.yaml"
REGISTRY_EXAMPLE = PROJECT_ROOT / "document_sources.example.yaml"
VERSIONS_PATH = PROJECT_ROOT / "document_versions.json"

# HTTP
ALLOWED_HOSTS = frozenset({
    "www.retsinformation.dk",
    "retsinformation.dk",
    "www.iaea.org",
    "iaea.org",
    "sst.dk",
    "www.sst.dk",
})
REQUEST_TIMEOUT = 15
MAX_BODY_SIZE = 2 * 1024 * 1024  # 2 MB

# retsinformation.dk eli/lta URL pattern: /eli/lta/YEAR/NR
_ELI_LTA_RE = re.compile(r"/eli/lta/(\d+)/(\d+)(?:/|$|\?)")


def _is_retsinformation_url(url: str) -> bool:
    """True if URL is for www.retsinformation.dk (not api subdomain)."""
    if not url:
        return False
    host = (urllib.parse.urlparse(url).netloc or "").lower()
    return "retsinformation.dk" in host and "api." not in host


@dataclass
class DocumentSource:
    id: str
    name: str
    url: str
    folder: str
    filename_hint: str | None
    version: str | None = None  # current version from document_sources.yaml (written after ingest)


@dataclass
class ResolvedUrl:
    """Normalized result for URL resolution steps."""

    label: str | None = None
    url: str | None = None


def load_registry_raw() -> list[dict[str, Any]]:
    """Load document_sources.yaml (or .example). Returns list of source dicts with id, name, url, folder, filename_hint. Includes sources with null url (e.g. local-only IAEA PDFs)."""
    path = REGISTRY_PATH if REGISTRY_PATH.exists() else REGISTRY_EXAMPLE
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sources = data.get("sources") or []
    return [s for s in sources if s.get("id") and s.get("name")]


def _load_registry() -> list[DocumentSource]:
    """Load registry as list of DocumentSource (uses load_registry_raw)."""
    raw = load_registry_raw()
    return [
        DocumentSource(
            id=s["id"],
            name=s["name"],
            url=(s.get("url") or "").strip(),
            folder=s.get("folder", "IAEA"),
            filename_hint=s.get("filename_hint"),
            version=(s.get("version") or "").strip() or None,
        )
        for s in raw
    ]


def _load_versions() -> dict[str, dict[str, str]]:
    if not VERSIONS_PATH.exists():
        return {}
    try:
        with open(VERSIONS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_versions(versions: dict[str, dict[str, str]]) -> None:
    with open(VERSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)


# Runtime caches (cleared per top-level update run)
_fetch_cache_lock = threading.Lock()
_fetch_cache: dict[str, str] = {}
_brave_cache_lock = threading.Lock()
_brave_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
_current_version_cache_lock = threading.Lock()
_current_version_cache: dict[str, str | None] = {}


def _reset_runtime_caches() -> None:
    """Clear ephemeral caches used within update/lookup runs."""
    with _fetch_cache_lock:
        _fetch_cache.clear()
    with _brave_cache_lock:
        _brave_cache.clear()
    with _current_version_cache_lock:
        _current_version_cache.clear()


def _resolve_sst_url_via_brave(source_name: str) -> str | None:
    """Use Brave Search to find sst.dk PDF URL. Returns first sst.dk PDF or any sst.dk link, or None."""
    q = f"site:sst.dk {source_name.strip()[:80]}".strip()
    if not q.replace("site:sst.dk", "").strip():
        return None
    first_any: str | None = None
    for r in _brave_search(q, count=10):
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or r.get("link") or "").strip()
        if not url or "sst.dk" not in url.lower():
            continue
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        if ".pdf" in url.lower():
            return url
        if first_any is None:
            first_any = url
    return first_any


def _is_sst_source(source: DocumentSource) -> bool:
    """True if this Danish source is hosted on sst.dk (vejledninger, not retsinformation.dk)."""
    if (source.url or "").strip():
        if "sst.dk" in source.url.lower():
            return True
    # Known SST document names (vejledninger)
    name_lower = (source.name or "").lower()
    if "åbne radioaktive" in name_lower or "aabne radioaktive" in name_lower or "radioaktive kilder" in name_lower:
        return True
    if "sikkerhedsvurdering" in name_lower:
        return True
    return False


def _resolve_danish_url_by_search(source_name: str, bek_nr: int | None = None) -> tuple[str | None, str | None]:
    """Find retsinformation.dk eli/lta URL via Brave (per-result titles allow title matching). Returns (label, url) or (None, None)."""
    if bek_nr is None and (not (source_name or "").strip() or not _danish_to_ascii_search(source_name or "")):
        return None, None
    url = _resolve_danish_url_via_brave(source_name, bek_nr=bek_nr)
    _brave_debug_log("by_search_brave", source_name=(source_name or "")[:60], bek_nr=bek_nr, got_url=bool(url), url=(url or "")[:80])
    if url:
        m = _ELI_LTA_RE.search(url)
        return f"BEK nr {m.group(2) if m else '?'} (search)", url
    return None, None


def _resolve_danish_url_to_newest(candidate_url: str) -> tuple[str | None, str | None]:
    """Given a retsinformation.dk eli/lta URL (e.g. 2019/670), fetch the page and parse 'Senere ændringer til forskriften'.
    Returns (newest_label, newest_url) — often a newer BEK number (e.g. 1385) that consolidates the original. If no amendments or parse fails, returns (None, None)."""
    if not _is_retsinformation_url(candidate_url):
        return None, None
    if not _allowed_url(candidate_url):
        return None, None
    try:
        html = _fetch_url(candidate_url)
    except Exception:
        return None, None
    return _parse_retsinformation(html, candidate_url)


def _resolve_danish_url_by_probing(bek_nr: int) -> tuple[str | None, str | None]:
    """Resolve Danish BEK URL by probing eli/lta/YEAR/nr for recent years. Returns (label, url) when page exists. No API needed.
    After finding a live URL, fetches that page and follows 'Senere ændringer' to return the newest consolidated version (e.g. 670 → 1385)."""
    base = "https://www.retsinformation.dk/eli/lta"
    current_year = date.today().year
    for year in range(current_year, current_year - 6, -1):
        url = f"{base}/{year}/{bek_nr}"
        if not _allowed_url(url):
            continue
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "RadiationSafetyRAG/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ssl.create_default_context()) as resp:
                if 200 <= resp.getcode() < 400:
                    # Page exists; follow "Senere ændringer" to get newest consolidated version (e.g. 670 → 1385)
                    newest_label, newest_url = _resolve_danish_url_to_newest(url)
                    if newest_label and newest_url:
                        return newest_label, newest_url
                    label = f"BEK nr {bek_nr} (probing {year})"
                    return label, url
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
    return None, None


def _extract_bek_number(source: DocumentSource) -> int | None:
    """Extract BEK number from source URL, version, current version file, or name/id (e.g. 'nr 670', 'BEK 670'). Returns e.g. 670 or None."""
    out = None
    # From URL: .../eli/lta/2019/670 or .../eli/lta/2025/1385
    if source.url:
        m = _ELI_LTA_RE.search(source.url)
        if m:
            out = int(m.group(2))
    if out is None:
        for version_str in (source.version, _get_current_version_from_file(source)):
            if version_str and isinstance(version_str, str):
                m = re.search(r"BEK\s+nr\s+(\d+)", version_str, re.IGNORECASE)
                if m:
                    out = int(m.group(1))
                    break
    if out is None:
        for raw in (source.name, source.id):
            if not raw or not isinstance(raw, str):
                continue
            m = re.search(r"(?:BEK\s+)?nr\s+(\d+)", raw, re.IGNORECASE)
            if m:
                out = int(m.group(1))
                break
            m = re.search(r"bek[-_]?(\d+)", raw, re.IGNORECASE)
            if m:
                out = int(m.group(1))
                break
    return out


def _extract_year_from_string(s: str) -> int | None:
    """Extract a 4-digit year (19xx or 20xx) from a string (version text or URL path). Returns None if none found."""
    if not (s or "").strip():
        return None
    m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
    return int(m.group(1)) if m else None


def _eli_lta_year_nr(url: str) -> tuple[int, int] | None:
    """Parse retsinformation.dk eli/lta URL to (year, nr). Returns None if not an eli/lta URL."""
    m = _ELI_LTA_RE.search(url or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _eli_lta_nr(url: str) -> int | None:
    """Parse retsinformation.dk eli/lta URL to BEK number."""
    m = _ELI_LTA_RE.search(url or "")
    return int(m.group(2)) if m else None


def _version_string_to_year_nr(version_str: str) -> tuple[int, int] | None:
    """Parse version history string to (year, nr). E.g. 'BEK nr 1385 af 18/11/2025' -> (2025, 1385)."""
    if not (version_str or "").strip():
        return None
    # "BEK nr 1385 af 18/11/2025" or "BEK nr 1385 af 18/11/25"
    m = re.search(r"BEK\s+nr\s+(\d+)\s+af\s+\d+/\d+/(\d{2,4})", version_str, re.IGNORECASE)
    if m:
        nr = int(m.group(1))
        year = int(m.group(2))
        if year < 100:
            year += 2000 if year < 50 else 1900
        return year, nr
    # "BEK nr 1384 (search)" with year elsewhere, e.g. "2025" in string
    m = re.search(r"BEK\s+nr\s+(\d+)", version_str, re.IGNORECASE)
    if m:
        nr = int(m.group(1))
        year_m = re.search(r"\b(20\d{2})\b", version_str)
        if year_m:
            return int(year_m.group(1)), nr
    return None


def _current_year_nr(source: DocumentSource, current_version: str | None) -> tuple[int, int] | None:
    """Our document's (year, nr): use the newer of URL and version history so a stale registry URL doesn't override version file."""
    from_url = _eli_lta_year_nr(source.url or "")
    from_version = _version_string_to_year_nr(current_version or "")
    if from_url and from_version:
        return from_version if from_version > from_url else from_url
    return from_url or from_version


def _reject_if_older_than_current(
    remote_label: str | None,
    download_url: str | None,
    current_url: str,
    current_version: str | None = None,
    source: DocumentSource | None = None,
) -> tuple[str | None, str | None]:
    """If we already have a newer version (from URL or version history) than the resolved one, return (None, None) so caller keeps looking."""
    if not download_url:
        return remote_label, download_url
    remote_yn = _eli_lta_year_nr(download_url)
    if not remote_yn:
        return remote_label, download_url
    # Our document is from (year, nr): URL first, then version history e.g. "BEK nr 1385 af 18/11/2025".
    current_yn = _current_year_nr(source, current_version) if source else None
    if not current_yn:
        current_yn = _eli_lta_year_nr(current_url or "")
    if not current_yn:
        return remote_label, download_url
    # Reject when remote is older or equal. (Same BEK number in another year = different decree; we only accept remotes that are title-matched, not from BEK-number-only search.)
    if remote_yn <= current_yn:
        return None, None
    return remote_label, download_url


def _apply_current_rejection(
    resolved: ResolvedUrl,
    *,
    current_url: str,
    current_version: str | None,
    source: DocumentSource,
    reject_older: bool,
) -> ResolvedUrl:
    if not reject_older:
        return resolved
    label, url = _reject_if_older_than_current(
        resolved.label,
        resolved.url,
        current_url=current_url,
        current_version=current_version,
        source=source,
    )
    return ResolvedUrl(label, url)


def _resolve_danish_source(
    source: DocumentSource,
    *,
    current_version: str | None = None,
    reject_older: bool = False,
) -> ResolvedUrl:
    """Resolve Danish source URL using unified fallback chain."""
    resolved = ResolvedUrl()
    bek_nr: int | None = None

    if _is_sst_source(source):
        sst_url = _resolve_sst_url_via_brave(source.name or source.id or "")
        if sst_url:
            resolved = ResolvedUrl(source.name or "SST vejledning", sst_url)
            resolved = _apply_current_rejection(
                resolved,
                current_url=source.url or "",
                current_version=current_version,
                source=source,
                reject_older=reject_older,
            )

    if not resolved.url:
        bek_nr = _extract_bek_number(source)
        if (source.name or "").strip():
            label, url = _resolve_danish_url_by_search(source.name, bek_nr=None)
            resolved = _apply_current_rejection(
                ResolvedUrl(label, url),
                current_url=source.url or "",
                current_version=current_version,
                source=source,
                reject_older=reject_older,
            )
        if not resolved.url and bek_nr is not None:
            label, url = _resolve_danish_url_by_probing(bek_nr)
            if url and _is_retsinformation_url(url):
                newest_label, newest_url = _resolve_danish_url_to_newest(url)
                if newest_label and newest_url:
                    label, url = newest_label, newest_url
            resolved = _apply_current_rejection(
                ResolvedUrl(label, url),
                current_url=source.url or "",
                current_version=current_version,
                source=source,
                reject_older=reject_older,
            )

    # For unnamed entries only: try BEK-number-based Brave + newest/probe fallback.
    if not resolved.url and bek_nr is not None and not (source.name or "").strip():
        label, url = _resolve_danish_url_by_search(source.name, bek_nr=bek_nr)
        if url:
            newest_label, newest_url = _resolve_danish_url_to_newest(url)
            if newest_label and newest_url:
                label, url = newest_label, newest_url
            else:
                bek_nr_from_url = _eli_lta_nr(url)
                if bek_nr_from_url is not None:
                    probe_label, probe_url = _resolve_danish_url_by_probing(bek_nr_from_url)
                    if probe_url:
                        label, url = probe_label, probe_url
        resolved = _apply_current_rejection(
            ResolvedUrl(label, url),
            current_url=source.url or "",
            current_version=current_version,
            source=source,
            reject_older=reject_older,
        )

    # Last fallback: parse source URL page for amendments.
    if not resolved.url and source.url:
        try:
            if _is_retsinformation_url(source.url):
                html = _fetch_url(source.url)
                label, url = _parse_retsinformation(html, source.url)
                resolved = ResolvedUrl(label, url)
        except Exception:
            pass

    return resolved


def _allowed_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.netloc or "").lower().lstrip("www.")
        if host in ALLOWED_HOSTS:
            return True
        if host.endswith(".dk") and "retsinformation" in url:
            return True
        return host in {"iaea.org", "www.iaea.org"}
    except Exception:
        return False


def _fetch_url(url: str) -> str:
    if not _allowed_url(url):
        raise ValueError(f"URL not allowlisted: {url}")
    with _fetch_cache_lock:
        cached = _fetch_cache.get(url)
    if cached is not None:
        return cached
    req = urllib.request.Request(url, headers={"User-Agent": "RadiationSafetyRAG/1.0"})
    # Prevent SSL issues on some systems
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ctx) as resp:
            if resp.headers.get("Content-Length") and int(resp.headers.get("Content-Length", 0)) > MAX_BODY_SIZE:
                raise ValueError("Response too large")
            data = resp.read(MAX_BODY_SIZE + 1)
            if len(data) > MAX_BODY_SIZE:
                raise ValueError("Response too large")
            decoded = data.decode("utf-8", errors="replace")
            with _fetch_cache_lock:
                _fetch_cache[url] = decoded
            return decoded
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code}: {url}")
    except urllib.error.URLError as e:
        raise ValueError(f"Request failed: {e.reason}")


def _parse_retsinformation(html: str, base_url: str) -> tuple[str | None, str | None]:
    """Parse 'Senere ændringer til forskriften' and return (newest_version_label, newest_url)."""
    base = base_url.split("/eli/")[0] if "/eli/" in base_url else "https://www.retsinformation.dk"
    # Normalize: single-quoted attributes to double-quoted so regex matches
    html = re.sub(r"\s+href='([^']+)'", r' href="\1"', html)
    matches = []

    # Pattern 1: <a href="/eli/lta/YYYY/NNN">...BEK nr NNN af D(M)/D(M)/YYYY</a>
    # Day/month can be 1 or 2 digits; allow optional nested tags and whitespace
    for pattern in (
        re.compile(
            r'href="(/eli/lta/(\d+)/(\d+))"[^>]*>\s*BEK\s+nr\s+\d+\s+af\s+(\d{1,2})/(\d{1,2})/(\d{4})',
            re.IGNORECASE,
        ),
        re.compile(
            r'href="(/eli/lta/(\d+)/(\d+))"[^>]*>(?:\s*<[^>]+>[^<]*)*\s*BEK\s+nr\s+\d+\s+af\s+(\d{1,2})/(\d{1,2})/(\d{4})',
            re.IGNORECASE,
        ),
        # Pattern 2: BEK nr NNN af D/M/YYYY anywhere, then find preceding href (same line/block)
        # Match href then up to 80 chars for the date
        re.compile(
            r'href="(/eli/lta/(\d+)/(\d+))"[^>]*>[\s\S]{0,80}?BEK\s+nr\s+(\d+)\s+af\s+(\d{1,2})/(\d{1,2})/(\d{4})',
            re.IGNORECASE,
        ),
    ):
        for m in pattern.finditer(html):
            g = m.groups()
            path = g[0]
            # Pattern 1/2: (path, year, nr, day, month, year) = 6 groups
            # Pattern 3: (path, year, nr, nr_text, day, month, year) = 7 groups
            if len(g) == 7:
                nr, day, month, year = g[3], g[4], g[5], g[6]
            elif len(g) == 6:
                nr, day, month, year = g[2], g[3], g[4], g[5]
            else:
                continue
            full_url = base + path if path.startswith("/") else base + "/" + path
            label = f"BEK nr {nr} af {day}/{month}/{year}"
            date_tuple = (int(year), int(month), int(day))
            matches.append((date_tuple, label, full_url))
        if matches:
            break

    # Fallback: find all /eli/lta/ links and all "BEK nr NNN af D/M/YYYY" in page, match by NNN
    if not matches:
        link_nrs: dict[str, str] = {}  # nr -> full url
        for path_m in re.finditer(r'href="(/eli/lta/(\d+)/(\d+))"', html):
            path, _year, nr = path_m.group(1), path_m.group(2), path_m.group(3)
            full_url = base + path if path.startswith("/") else base + "/" + path
            link_nrs[nr] = full_url
        for bek_m in re.finditer(
            r'BEK\s+nr\s+(\d+)\s+af\s+(\d{1,2})/(\d{1,2})/(\d{4})',
            html,
            re.IGNORECASE,
        ):
            nr, day, month, year = bek_m.groups()
            if nr in link_nrs:
                label = f"BEK nr {nr} af {day}/{month}/{year}"
                date_tuple = (int(year), int(month), int(day))
                matches.append((date_tuple, label, link_nrs[nr]))

    if not matches:
        return None, None
    matches.sort(key=lambda x: x[0])
    _, label, url = matches[-1]
    return label, url


# IAEA search: base URL and rate limit (avoid hammering)
IAEA_SEARCH_BASE = "https://www.iaea.org/publications/search"
_LAST_IAEA_SEARCH: float = 0
IAEA_SEARCH_DELAY_SEC = 2.0


def _lookup_iaea_publication_url(query: str) -> str | None:
    """Fetch IAEA publications search with query and return first publication page URL (https://www.iaea.org/publications/ID/slug), or None."""
    global _LAST_IAEA_SEARCH
    now = time.monotonic()
    if now - _LAST_IAEA_SEARCH < IAEA_SEARCH_DELAY_SEC:
        time.sleep(IAEA_SEARCH_DELAY_SEC - (now - _LAST_IAEA_SEARCH))
    _LAST_IAEA_SEARCH = time.monotonic()
    search_query = (query or "").strip()[:80]
    if not search_query:
        return None
    url = f"{IAEA_SEARCH_BASE}?keywords={urllib.parse.quote(search_query, safe='')}"
    if not _allowed_url(url):
        return None
    try:
        html = _fetch_url(url)
    except Exception:
        return None
    # Normalize single-quoted hrefs to double-quoted for one regex pass
    html = re.sub(r"href='([^']+)'", r'href="\1"', html)
    # Match full URL or relative /publications/ID/slug
    for m in re.finditer(
        r'href="(https://www\.iaea\.org/publications/(\d+)/[^"]+)"',
        html,
    ):
        return m.group(1)
    for m in re.finditer(
        r'href="(/publications/(\d+)/[^"]+)"',
        html,
    ):
        return f"https://www.iaea.org{m.group(1)}"
    return None


def _danish_to_ascii_search(s: str) -> str:
    """Replace Danish letters with ASCII equivalents for search queries to avoid encoding issues."""
    if not s:
        return s
    s = s.strip()[:80]
    for old, new in (("å", "a"), ("ø", "o"), ("æ", "ae"), ("Å", "A"), ("Ø", "O"), ("Æ", "Ae")):
        s = s.replace(old, new)
    return s


def _brave_search(query: str, count: int = 10) -> list[dict[str, Any]]:
    """Call Brave Search API; return list of result dicts with url/link and title. Returns [] on error or no key."""
    api_key = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    if not api_key:
        return []
    cache_key = (query, count)
    with _brave_cache_lock:
        cached = _brave_cache.get(cache_key)
    if cached is not None:
        return cached
    api_url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count={count}"
    req = urllib.request.Request(
        api_url,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key, "User-Agent": "RadiationSafetyRAG/1.0"},
    )
    _brave_throttle()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ssl.create_default_context()) as resp:
            data = json.load(resp)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    web = data.get("web") or data
    results = web.get("results") or web.get("web") or []
    if not isinstance(results, list):
        return []
    with _brave_cache_lock:
        _brave_cache[cache_key] = results
    return results


def _brave_debug_log(msg: str, **kwargs: object) -> None:
    """Append one line to brave_search_debug.log for diagnosing Brave search. Only when BRAVE_DEBUG=1."""
    if os.getenv("BRAVE_DEBUG", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        line = json.dumps({"ts": datetime.now().isoformat(), "msg": msg, **kwargs}, ensure_ascii=False) + "\n"
        with (PROJECT_ROOT / "brave_search_debug.log").open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _resolve_danish_url_via_brave(source_name: str, bek_nr: int | None = None) -> str | None:
    """Use Brave Search to find retsinformation.dk eli/lta URL. Filter by title, exclude Historisk. Returns newest match or None."""
    if bek_nr is not None:
        q = f"site:retsinformation.dk BEK {bek_nr}"
    else:
        name_part = _danish_to_ascii_search(source_name or "")
        if not name_part:
            _brave_debug_log("brave_skip", reason="empty_name_part")
            return None
        q = f"site:retsinformation.dk {name_part}"
    _brave_debug_log("brave_called", source_name=(source_name or "")[:80], bek_nr=bek_nr, has_api_key=bool(os.getenv("BRAVE_SEARCH_API_KEY")))
    results = _brave_search(q, count=15)
    _brave_debug_log("brave_results_count", count=len(results))
    title_norm = (_danish_to_ascii_search(source_name or "")).lower().strip()

    def _is_historisk(result_title: str, result_url: str) -> bool:
        """Exclude historical/archived versions; we want the current decree."""
        t = (_danish_to_ascii_search((result_title or "").strip())).lower()
        u = (result_url or "").lower()
        return "historisk" in t or "historisk" in u

    def _title_matches(result_title: str) -> bool:
        if not title_norm:
            return True
        t = (_danish_to_ascii_search((result_title or "").strip())).lower()
        if not t:
            return False
        if title_norm in t:
            return True
        if len(title_norm) > 25:
            suffix = title_norm[-40:].strip()
            if len(suffix) >= 15 and suffix in t:
                return True
        # Same decree under a variant name: e.g. "Bekendtgørelse om radioaktive stoffer" vs "Bekendtgørelse om brug af radioaktive stoffer"
        if "radioaktive stoffer" in title_norm and "radioaktive stoffer" in t and "transport" not in t:
            return True
        # "Brug af åbne radioaktive kilder" vs "Bekendtgørelse om anvendelse af åbne radioaktive kilder" (same decree)
        if "abne radioaktive kilder" in title_norm and "abne radioaktive kilder" in t and "lukkede" not in t:
            return True
        return False

    candidates: list[tuple[int, int, str]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or r.get("link") or "").strip()
        if not url:
            continue
        m = _ELI_LTA_RE.search(url)
        if not m:
            continue
        result_title = (r.get("title") or r.get("name") or "").strip()
        if _is_historisk(result_title, url):
            _brave_debug_log("brave_result", url=url[:90], title=(result_title or "")[:100], title_matched=False, skipped="historisk")
            continue
        matched = _title_matches(result_title)
        _brave_debug_log("brave_result", url=url[:90], title=(result_title or "")[:100], title_matched=matched)
        if not matched:
            continue
        year, nr = int(m.group(1)), int(m.group(2))
        full_url = url if url.startswith("http") else f"https://{url}"
        candidates.append((year, nr, full_url))
    if not candidates:
        _brave_debug_log("brave_return", returned=None, reason="no_candidates_after_title_filter")
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = candidates[0][2]
    _brave_debug_log("brave_return", returned=chosen[:90], candidates_count=len(candidates))
    return chosen


def _lookup_iaea_publication_url_via_brave(query: str) -> str | None:
    """Use Brave Search to find IAEA publication URL. Returns first iaea.org/publications/ID/slug or None."""
    q = f"site:iaea.org/publications {query}".strip()[:100]
    if not q.replace("site:iaea.org/publications", "").strip():
        return None
    for r in _brave_search(q, count=10):
        if isinstance(r, dict):
            url = (r.get("url") or r.get("link") or "").strip()
            if url and re.match(r"https?://www\.iaea\.org/publications/\d+/", url):
                return url if url.startswith("http") else f"https://{url}"
    return None


def _lookup_iaea_publication_url_multi(queries: list[str]) -> str | None:
    """Try each query via IAEA search; if none found, try Brave Search (if key set) for first few terms. Returns first URL found."""
    for q in queries:
        if not (q or "").strip():
            continue
        url = _lookup_iaea_publication_url(q)
        if url:
            return url
    # Fallback: Brave Search (site:iaea.org/publications) for first 3 non-empty terms
    for q in queries[:3]:
        if not (q or "").strip():
            continue
        url = _lookup_iaea_publication_url_via_brave(q)
        if url:
            return url
    return None


def _parse_iaea_superseded(html: str) -> tuple[str | None, str | None]:
    """Parse 'Superseded by: ...' and return (superseding_title, superseding_url)."""
    # Superseded by: <a href="...">Specific Safety Guide - SSG-20 (Rev. 1)</a>
    m = re.search(
        r'Superseded\s+by\s*:\s*<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        url, title = m.group(1), m.group(2).strip()
        if url.startswith("/"):
            url = "https://www.iaea.org" + url
        return title, url
    # Fallback: Superseded by: Some text (maybe without link)
    m = re.search(r'Superseded\s+by\s*:\s*([^\n<]+)', html, re.IGNORECASE)
    if m:
        return m.group(1).strip(), None
    return None, None


def get_local_pdf_path(source: DocumentSource) -> Path | None:
    """Return path to the local PDF for this source, or None if not found."""
    folder_path = DOCS_DIR / source.folder
    if not folder_path.exists():
        return None
    pdfs = list(folder_path.rglob("*.pdf"))
    if source.filename_hint:
        for p in sorted(pdfs):
            if source.filename_hint in p.name:
                return p
        return None
    if len(pdfs) == 1:
        return pdfs[0]
    return None


def _get_current_version_from_file(source: DocumentSource) -> str | None:
    """Infer current version from disk: version file (Danish), or file mtime (filename_hint)."""
    cache_key = f"{DOCS_DIR}|{source.id}|{source.folder}|{source.filename_hint or ''}"
    with _current_version_cache_lock:
        if cache_key in _current_version_cache:
            return _current_version_cache[cache_key]

    folder_path = DOCS_DIR / source.folder
    if not folder_path.exists():
        with _current_version_cache_lock:
            _current_version_cache[cache_key] = None
        return None
    # Danish: ingestion writes {source_id}_version.txt with the label we ingested
    if source.folder == "Bekendtgørelse":
        version_file = folder_path / f"{source.id}_version.txt"
        if version_file.exists():
            try:
                v = version_file.read_text(encoding="utf-8").strip()
                with _current_version_cache_lock:
                    _current_version_cache[cache_key] = v
                return v
            except OSError:
                pass
    # Fallback: match by filename_hint and use mtime as date
    if source.filename_hint:
        for p in folder_path.rglob("*"):
            if source.filename_hint in p.name:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                v = mtime.strftime("%Y-%m-%d")
                with _current_version_cache_lock:
                    _current_version_cache[cache_key] = v
                return v
    with _current_version_cache_lock:
        _current_version_cache[cache_key] = None
    return None


def check_one_source(
    source: DocumentSource,
    versions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Check a single source for updates. Returns dict for API response."""
    current = (
        versions.get(source.id, {}).get("version")
        or source.version
        or _get_current_version_from_file(source)
    )
    local_path = get_local_pdf_path(source)
    result = {
        "id": source.id,
        "name": source.name,
        "url": source.url,
        "folder": source.folder,
        "current_version": current,
        "remote_version": None,
        "update_available": False,
        "local_date": None,
        "remote_date": None,
        "download_url": source.url,
        "has_local_file": local_path is not None,
        "error": None,
    }

    # Danish (Bekendtgørelse): sst.dk (Brave search) or retsinformation.dk (probe + Brave search + "Senere ændringer")
    is_danish = (source.folder or "").strip() == "Bekendtgørelse"
    if is_danish:
        resolved = _resolve_danish_source(source, current_version=current, reject_older=True)
        remote_label, download_url = resolved.label, resolved.url
        if remote_label and download_url:
            result["remote_version"] = remote_label
            result["download_url"] = download_url
            current_yn = _eli_lta_year_nr(source.url or "")
            remote_yn = _eli_lta_year_nr(download_url or "")
            if current_yn and remote_yn:
                result["update_available"] = remote_yn > current_yn
            else:
                # SST or other: compare years from version/URL so we notice "our 2020" vs "remote 2021"
                our_year = _extract_year_from_string(current or "") or _extract_year_from_string(source.url or "")
                remote_year = _extract_year_from_string(download_url or "")
                if our_year and remote_year:
                    result["update_available"] = remote_year > our_year
                else:
                    result["update_available"] = (download_url or "").strip() != (source.url or "").strip()
        else:
            result["remote_version"] = (
                "Local only (no URL)" if not source.url
                else "Cannot detect newest (amendments may be JS-rendered). Check Senere aendringer."
            )
            result["download_url"] = source.url or ""
        return result

    if not source.url:
        result["remote_version"] = "Local only (no URL)"
        result["download_url"] = ""
        return result

    try:
        if _is_retsinformation_url(source.url):
            # Prefer Retsinformation harvest API if we have documents
            remote_label, download_url = None, None
            # Fetch HTML and parse "Senere ændringer" (amendments list may be JS-rendered)
            if not (remote_label and download_url):
                html = _fetch_url(source.url)
                remote_label, download_url = _parse_retsinformation(html, source.url)
            if remote_label and download_url:
                result["remote_version"] = remote_label
                result["download_url"] = download_url
                # Only show update when remote URL is different from what we have
                result["update_available"] = (download_url or "").strip() != (source.url or "").strip()
            else:
                # Amendments list missing (often because the site loads it via JavaScript)
                result["remote_version"] = (
                    "Cannot detect newest (amendments list not in page—site may load it via JavaScript). "
                    "Open the link and check “Senere ændringer til forskriften” for the latest BEK."
                )
                result["download_url"] = source.url

        elif "iaea.org" in (urllib.parse.urlparse(source.url).netloc or "").lower():
            html = _fetch_url(source.url)
            superseded_title, superseded_url = _parse_iaea_superseded(html)
            if superseded_title and superseded_url:
                result["remote_version"] = superseded_title
                result["download_url"] = superseded_url
                result["update_available"] = (current != superseded_title) if current else True
            else:
                # Publication is current (no superseding edition)
                result["remote_version"] = "Current"
                result["download_url"] = source.url
                result["update_available"] = False

        else:
            # Generic: HEAD request for direct PDF or last-modified
            req = urllib.request.Request(source.url, method="HEAD", headers={"User-Agent": "RadiationSafetyRAG/1.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ssl.create_default_context()) as resp:
                lm = resp.headers.get("Last-Modified")
                if lm:
                    result["remote_version"] = lm
                    result["remote_date"] = lm
                result["download_url"] = source.url
                if not current:
                    result["update_available"] = True
                elif lm:
                    result["update_available"] = True  # Conservative: assume update if we have a date

    except Exception as e:
        result["error"] = str(e)
        result["remote_version"] = None
        result["update_available"] = False

    return result


def check_updates() -> list[dict[str, Any]]:
    """Load registry and version state, check each source (one Brave request per document, in parallel), return list of status dicts."""
    _reset_runtime_caches()
    registry = _load_registry()
    versions = _load_versions()
    # One request per document; run checks in parallel with bounded concurrency to avoid rate limits.
    max_workers = min(4, max(1, len(registry)))
    results_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(check_one_source, s, versions): i for i, s in enumerate(registry)}
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results_by_index[i] = future.result()
            except Exception as e:
                results_by_index[i] = {
                    "id": registry[i].id,
                    "name": registry[i].name,
                    "error": str(e),
                    "update_available": False,
                }
    return [results_by_index[i] for i in range(len(registry))]


def update_version_after_ingest(source_id: str, version: str) -> None:
    """Write current version for a source (e.g. after user re-ingests). Updates document_versions.json and document_sources.yaml."""
    versions = _load_versions()
    versions[source_id] = {
        "version": version,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _save_versions(versions)
    update_registry_version(source_id, version)


def _update_registry_field(source_id: str, field: str, value: str) -> None:
    """Update one field for a source in document_sources.yaml."""
    if not REGISTRY_PATH.exists():
        return
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for s in data.get("sources") or []:
        if isinstance(s, dict) and (s.get("id") or "").strip() == source_id.strip():
            s[field] = value.strip()
            break
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def update_registry_version(source_id: str, version: str) -> None:
    """Write current version for a source into document_sources.yaml."""
    _update_registry_field(source_id, "version", version)


def lookup_source_url(source_id: str) -> tuple[str | None, str | None]:
    """Try to find the document URL for a source (Danish: sst.dk or retsinformation.dk via Brave/probe; IAEA: search). Returns (url, error_message)."""
    _reset_runtime_caches()
    registry = _load_registry()
    source = next((s for s in registry if (s.id or "").strip() == source_id.strip()), None)
    if not source:
        return None, "Source not found"

    folder = (source.folder or "").strip()
    name = (source.name or source.id or "").strip()

    if folder == "Bekendtgørelse":
        resolved = _resolve_danish_source(source, current_version=None, reject_older=False)
        if resolved.url:
            return resolved.url, None
        return None, "Could not find URL from retsinformation.dk (probe or search)."

    if folder in ("IAEA", "IAEA_other"):
        queries = [name] if name else [source_id]
        url = _lookup_iaea_publication_url_multi(queries)
        if url:
            return url, None
        return None, "Could not find URL on iaea.org."

    return None, "Unknown folder; only Bekendtgørelse and IAEA/IAEA_other are supported."


def update_registry_url(source_id: str, new_url: str) -> None:
    """Update document_sources.yaml so the source's url is set to new_url (e.g. after resolving newest Danish)."""
    _update_registry_field(source_id, "url", new_url)


def append_source_to_registry(
    source_id: str,
    name: str,
    *,
    url: str | None = None,
    folder: str = "IAEA_other",
    filename_hint: str | None = None,
    version: str | None = None,
) -> None:
    """Append a new source to document_sources.yaml. Ensures source_id is unique by appending -1, -2 if needed."""
    data: dict[str, Any] = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    sources: list[dict[str, Any]] = list(data.get("sources") or [])
    existing_ids = {s.get("id") for s in sources if isinstance(s, dict) and s.get("id")}
    sid = source_id
    c = 0
    while sid in existing_ids:
        c += 1
        sid = f"{source_id}-{c}"
        existing_ids.add(sid)
    entry = {
        "id": sid,
        "name": name,
        "url": (url or "").strip() or None,
        "folder": folder,
        "filename_hint": filename_hint,
        "version": version,
    }
    sources.append(entry)
    data["sources"] = sources
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
