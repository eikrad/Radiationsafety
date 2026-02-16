"""
Discover local documents (PDFs + Danish version files), extract version/title from PDFs,
optionally confirm URLs on IAEA/retsinformation, and write document_sources.yaml with all current documents.
Run: uv run python build_document_sources.py
"""

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "documents"
REGISTRY_PATH = PROJECT_ROOT / "document_sources.yaml"
REGISTRY_EXAMPLE = PROJECT_ROOT / "document_sources.example.yaml"


def _slug(s: str) -> str:
    """Make a short id slug from a title (e.g. for IAEA sources)."""
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s).strip().lower()
    return s[:48] or "doc"


def _extract_pdf_title_and_version(pdf_path: Path) -> tuple[str | None, str | None]:
    """Extract title and version-like string from PDF metadata and first page text. Returns (title, version)."""
    title: str | None = None
    version: str | None = None
    try:
        reader = PdfReader(str(pdf_path))
        if reader.metadata is not None:
            title = getattr(reader.metadata, "title", None)
            if isinstance(title, str):
                title = title.strip() or None
        if reader.pages:
            page0 = reader.pages[0]
            text = page0.extract_text() or ""
            # IAEA: common patterns on first page
            for pattern in [
                r"Safety\s+Reports\s+Series\s+No\.\s*(\d+)",
                r"Safety\s+Standards\s+Series\s+No\.\s*([\w-]+)",
                r"(SSR-\d+(?:\s*/\s*\d+)?(?:\s*\(Rev\.\s*\d+\))?)",
                r"(SSG-\d+(?:\s*\(Rev\.\s*\d+\))?)",
                r"(No\.\s+SS[RGP][-\s]\d+[^.\n]{0,40})",
                r"IAEA[\s-](\w+)[\s-](\d+)",
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    version = m.group(0).strip()[:80]
                    if not title:
                        title = version
                    break
            if not version and not title and len(text) > 20:
                first_line = text.split("\n")[0].strip()[:100]
                if first_line:
                    title = first_line
    except Exception:
        pass
    if not title:
        title = pdf_path.stem.replace("_", " ").replace("-", " ")
    return title, version or title


def _extract_iaea_search_terms(pdf_path: Path) -> list[str]:
    """Extract an ordered list of search terms from a PDF for IAEA URL lookup (STI/PUB, TECDOC, series, title words)."""
    seen: set[str] = set()
    terms: list[str] = []
    title, version = _extract_pdf_title_and_version(pdf_path)
    text = ""
    try:
        reader = PdfReader(str(pdf_path))
        if reader.metadata is not None:
            for attr in ("title", "subject", "keywords", "creator"):
                v = getattr(reader.metadata, attr, None)
                if isinstance(v, str) and len(v) > 2:
                    text += " " + v
        if reader.pages:
            text += " " + (reader.pages[0].extract_text() or "")
    except Exception:
        pass
    text = " " + (version or "") + " " + (title or "") + " " + pdf_path.stem

    def add(s: str) -> None:
        s = (s or "").strip()[:80]
        if s and s not in seen:
            seen.add(s)
            terms.append(s)

    # STI/PUB/1234 or STI/PUB/1234-VOL2 (and space variant for search engines that split on /)
    for m in re.finditer(r"STI/PUB/\s*(\d+(?:-\w+)?)", text, re.IGNORECASE):
        num = m.group(1)
        add("STI/PUB/" + num)
        add("STI PUB " + num)
    # IAEA-TECDOC-1234 or TECDOC-1234 or TECDOC 1234
    for m in re.finditer(r"(?:IAEA-)?TECDOC[- ](\d+)", text, re.IGNORECASE):
        add("TECDOC " + m.group(1))
    for m in re.finditer(r"IAEA-TECDOC-(\d+)", text, re.IGNORECASE):
        add("IAEA-TECDOC-" + m.group(1))
    # Series: SSG-20, SSR-3, SSG-20 (Rev. 1), No. SSG-20
    for m in re.finditer(r"(SS[RGP][-\s]?\d+(?:\s*\(Rev\.\s*\d+\))?)", text, re.IGNORECASE):
        add(m.group(1).strip())
    # ISBN (IAEA often 92-0-xxxxxx-x)
    for m in re.finditer(r"92-0-\d[\d-]{8,14}[Xx\d]", text):
        add(m.group(0))
    # Version/title from extract (if not already added)
    if version:
        add(version)
    if title:
        add(title)
        words = [w for w in re.split(r"\W+", title) if len(w) > 2][:5]
        if len(words) > 2:
            add(" ".join(words))
    if not terms:
        add(pdf_path.stem.replace("_", " ").replace("-", " "))
    return terms


def _discover_iaea_pdfs() -> list[dict[str, Any]]:
    """Find all PDFs in IAEA and IAEA_other; extract title and version from each."""
    out: list[dict[str, Any]] = []
    for folder in ("IAEA", "IAEA_other"):
        base = DOCS_DIR / folder
        if not base.exists():
            continue
        for pdf_path in sorted(base.rglob("*.pdf")):
            rel = pdf_path.relative_to(base)
            title, version = _extract_pdf_title_and_version(pdf_path)
            slug = _slug(title or pdf_path.stem)
            source_id = f"iaea-{slug}" if folder == "IAEA" else f"iaea-other-{slug}"
            out.append({
                "id": source_id,
                "name": title or pdf_path.stem,
                "url": None,
                "folder": folder,
                "filename_hint": pdf_path.name,
                "version": version or None,
                "_path": str(pdf_path),
            })
    return out


def _discover_danish_from_version_files() -> list[dict[str, Any]]:
    """Discover Danish sources from Bekendtgørelse: *_version.txt and optionally *_current.xml."""
    out: list[dict[str, Any]] = []
    bek = DOCS_DIR / "Bekendtgørelse"
    if not bek.exists():
        return out
    for version_file in sorted(bek.glob("*_version.txt")):
        source_id = version_file.stem.replace("_version", "")
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            version = None
        name = "Radioaktivitetsbekendtgørelsen" if "radioaktiv" in source_id.lower() else source_id
        # Build ELI URL from version string "BEK nr NNN af DD/MM/YYYY" -> .../eli/lta/YYYY/NNN
        url = None
        if version:
            m = re.search(r"BEK\s+nr\s+(\d+)\s+af\s+\d{1,2}/\d{1,2}/(\d{4})", version, re.IGNORECASE)
            if m:
                nr, year = m.group(1), m.group(2)
                url = f"https://www.retsinformation.dk/eli/lta/{year}/{nr}"
        out.append({
            "id": source_id,
            "name": name,
            "url": url,
            "folder": "Bekendtgørelse",
            "filename_hint": None,
            "version": version,
            "_path": str(version_file),
        })
    return out


def _discover_danish_pdfs() -> list[dict[str, Any]]:
    """Discover Danish PDFs in Bekendtgørelse (not yet covered by _version.txt)."""
    out: list[dict[str, Any]] = []
    bek = DOCS_DIR / "Bekendtgørelse"
    if not bek.exists():
        return out
    seen_ids = {s["id"] for s in _discover_danish_from_version_files()}
    for pdf_path in sorted(bek.rglob("*.pdf")):
        title, version = _extract_pdf_title_and_version(pdf_path)
        slug = _slug(title or pdf_path.stem)
        source_id = f"dk-{slug}"
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        # Try to get ELI from "BEK nr NNN af ..." in version
        url = None
        if version:
            m = re.search(r"BEK\s+nr\s+(\d+)\s+af\s+\d{1,2}/\d{1,2}/(\d{4})", version, re.IGNORECASE)
            if m:
                nr, year = m.group(1), m.group(2)
                url = f"https://www.retsinformation.dk/eli/lta/{year}/{nr}"
        out.append({
            "id": source_id,
            "name": title or pdf_path.stem,
            "url": url,
            "folder": "Bekendtgørelse",
            "filename_hint": pdf_path.name,
            "version": version,
            "_path": str(pdf_path),
        })
    return out


def _load_existing_registry() -> list[dict[str, Any]]:
    """Load existing document_sources.yaml (or example) to reuse URLs and merge with discoveries."""
    path = REGISTRY_PATH if REGISTRY_PATH.exists() else REGISTRY_EXAMPLE
    if not path.exists():
        return []
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("sources") or []


def _merge_url_and_version(discovered: dict[str, Any], existing: list[dict[str, Any]]) -> dict[str, Any]:
    """Fill url from existing registry if we find a match by id or name; keep version from discovery."""
    out = {k: v for k, v in discovered.items() if not k.startswith("_")}
    for ex in existing:
        if not isinstance(ex, dict):
            continue
        if ex.get("id") == discovered.get("id"):
            out["url"] = ex.get("url") or out.get("url")
            if not out.get("version") and ex.get("version"):
                out["version"] = ex.get("version")
            break
        if ex.get("name") and discovered.get("name") and ex.get("name").strip() == discovered.get("name", "").strip():
            out["url"] = ex.get("url") or out.get("url")
            break
    return out


def _confirm_danish_url(source: dict[str, Any]) -> None:
    """Optionally confirm Danish URL by fetching the page and checking for newest (leave url as-is if already set)."""
    if not source.get("url") or "retsinformation" not in (source.get("url") or ""):
        return
    try:
        from document_updates import _fetch_url, _parse_retsinformation
        html = _fetch_url(source["url"])
        label, new_url = _parse_retsinformation(html, source["url"])
        if new_url and new_url != source["url"]:
            source["url"] = new_url
            if label:
                source["version"] = label
    except Exception:
        pass


def _confirm_iaea_url(source: dict[str, Any]) -> None:
    """If IAEA source has no URL, try to look up publication page via IAEA search (multiple terms from PDF)."""
    if source.get("url") or source.get("folder") not in ("IAEA", "IAEA_other"):
        return
    try:
        from document_updates import _lookup_iaea_publication_url_multi
        terms = []
        path = source.get("_path")
        if path:
            terms = _extract_iaea_search_terms(Path(path))
        if not terms:
            version = (source.get("version") or "").strip()
            name = (source.get("name") or "").strip()
            if version:
                terms.append(version)
            if name and name not in terms:
                terms.append(name)
        if terms:
            url = _lookup_iaea_publication_url_multi(terms)
            if url:
                source["url"] = url
    except Exception:
        pass


def build_sources(
    *,
    confirm_urls: bool = True,
) -> list[dict[str, Any]]:
    """Discover all local documents, extract versions, optionally confirm URLs, merge with existing registry. Returns list of source dicts for YAML."""
    existing = _load_existing_registry()
    sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Danish from version files (ingested XML)
    for d in _discover_danish_from_version_files():
        d = _merge_url_and_version(d, existing)
        if confirm_urls:
            _confirm_danish_url(d)
        sid = d.get("id")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            sources.append(d)

    # Danish PDFs
    for d in _discover_danish_pdfs():
        d = _merge_url_and_version(d, existing)
        if confirm_urls and d.get("url"):
            _confirm_danish_url(d)
        sid = d.get("id")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            sources.append(d)

    # IAEA PDFs (deduplicate ids by appending suffix)
    for d in _discover_iaea_pdfs():
        d = _merge_url_and_version(d, existing)
        if confirm_urls:
            _confirm_iaea_url(d)
        sid = d.get("id")
        if not sid:
            continue
        base_id = sid
        c = 0
        while sid in seen_ids:
            c += 1
            sid = f"{base_id}-{c}"
        d["id"] = sid
        seen_ids.add(sid)
        sources.append(d)

    # Normalise for YAML: no _path, version can be null
    result = []
    for s in sources:
        row = {
            "id": s.get("id", ""),
            "name": s.get("name", "Unknown"),
            "url": s.get("url") or None,
            "folder": s.get("folder", "IAEA"),
            "filename_hint": s.get("filename_hint"),
            "version": s.get("version"),
        }
        result.append(row)
    return result


def write_document_sources_yaml(sources: list[dict[str, Any]], path: Path | None = None) -> None:
    """Write sources to document_sources.yaml (or given path)."""
    import yaml
    path = path or REGISTRY_PATH
    data = {
        "sources": sources,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Extract versions from local PDFs and build document_sources.yaml")
    p.add_argument("--no-confirm", action="store_true", help="Skip looking up URLs on websites")
    p.add_argument("--dry-run", action="store_true", help="Print sources but do not write YAML")
    p.add_argument("-o", "--output", default=None, help="Output YAML path (default: document_sources.yaml)")
    args = p.parse_args()
    sources = build_sources(confirm_urls=not args.no_confirm)
    if not sources:
        print("No documents discovered. Add PDFs to documents/IAEA, documents/IAEA_other, or documents/Bekendtgørelse.")
        return
    print(f"Discovered {len(sources)} document source(s):")
    for s in sources:
        print(f"  - {s['id']}: {s['name']} (version: {s.get('version') or '—'})")
    if args.dry_run:
        print("(Dry run: not writing file)")
        return
    out_path = Path(args.output) if args.output else REGISTRY_PATH
    write_document_sources_yaml(sources, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
