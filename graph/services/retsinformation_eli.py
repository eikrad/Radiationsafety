"""Resolve latest Danish legislation via ELI metadata relations."""

from __future__ import annotations

import json
import re
import ssl
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Any

RELATION_KEYS = (
    "changed_by",
    "consolidated_by",
    "changes",
    "consolidates",
    "is_changed_by",
    "is_consolidated_by",
)
PREFERRED_FORWARD_KEYS = (
    "changed_by",
    "consolidated_by",
    "is_changed_by",
    "is_consolidated_by",
)
_ISSUE_DATE_RE = re.compile(
    r"BEK\s+nr\s+(\d+)\s+af\s+(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)


@dataclass
class EliEdge:
    relation: str
    from_url: str
    to_url: str


@dataclass
class EliNode:
    url: str
    label: str | None = None
    issue_date: date | None = None
    year_nr: tuple[int, int] | None = None


@dataclass
class EliResolution:
    start_url: str
    chosen_url: str | None
    chosen_label: str | None
    confidence: float
    visited_urls: list[str]
    edges: list[EliEdge]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_url": self.start_url,
            "chosen_url": self.chosen_url,
            "chosen_label": self.chosen_label,
            "confidence": self.confidence,
            "visited_urls": self.visited_urls,
            "edges": [e.__dict__ for e in self.edges],
            "reason": self.reason,
        }


def _canonical_metadata_url(eli_url: str) -> str:
    base = (eli_url or "").strip()
    if not base:
        return base
    if base.endswith(".json"):
        return base
    return f"{base.rstrip('/')}.json"


def _extract_label(meta: dict[str, Any]) -> str | None:
    candidates = (
        "title",
        "label",
        "name",
        "eli:title",
        "dct:title",
        "documentTitle",
        "overskrift",
    )
    for key in candidates:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_year_nr(url: str) -> tuple[int, int] | None:
    m = re.search(r"/eli/lta/(\d+)/(\d+)(?:/|$|\?)", url or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_issue_date(label: str | None) -> date | None:
    if not (label or "").strip():
        return None
    m = _ISSUE_DATE_RE.search(label or "")
    if not m:
        return None
    day = int(m.group(2))
    month = int(m.group(3))
    year = int(m.group(4))
    if year < 100:
        year += 2000 if year < 50 else 1900
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _fetch_json(url: str) -> dict[str, Any] | None:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "RadiationSafetyRAG/1.0"},
    )
    with urllib.request.urlopen(
        req, timeout=20, context=ssl.create_default_context()
    ) as resp:
        data = json.load(resp)
    if isinstance(data, dict):
        return data
    return None


def _iter_relation_urls(payload: Any, relation_key: str) -> list[str]:
    urls: list[str] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            key_norm = str(k).lower().replace(":", "_")
            if key_norm.endswith(relation_key):
                if isinstance(v, str) and v.startswith("http"):
                    urls.append(v.strip())
                elif isinstance(v, dict):
                    for nested_key in ("@id", "id", "href", "url", "iri"):
                        nested_val = v.get(nested_key)
                        if isinstance(nested_val, str) and nested_val.startswith(
                            "http"
                        ):
                            urls.append(nested_val.strip())
                elif isinstance(v, list):
                    for item in v:
                        urls.extend(
                            _iter_relation_urls({relation_key: item}, relation_key)
                        )
            urls.extend(_iter_relation_urls(v, relation_key))
    elif isinstance(payload, list):
        for item in payload:
            urls.extend(_iter_relation_urls(item, relation_key))
    return urls


def _extract_forward_urls(meta: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key in PREFERRED_FORWARD_KEYS:
        urls = _iter_relation_urls(meta, key)
        for url in urls:
            out.append((key, url))
    # de-duplicate while preserving priority order
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for rel, url in out:
        if url in seen:
            continue
        seen.add(url)
        deduped.append((rel, url))
    return deduped


def _best_node(nodes: list[EliNode]) -> EliNode | None:
    if not nodes:
        return None
    ranked = sorted(
        nodes,
        key=lambda n: (
            n.issue_date is not None,
            n.issue_date or date.min,
            n.year_nr or (0, 0),
        ),
        reverse=True,
    )
    return ranked[0]


def resolve_latest_document(start_url: str, max_depth: int = 12) -> EliResolution:
    """Traverse ELI relations to resolve a latest known successor."""
    start = (start_url or "").strip()
    if not start:
        return EliResolution(
            start_url=start_url,
            chosen_url=None,
            chosen_label=None,
            confidence=0.0,
            visited_urls=[],
            edges=[],
            reason="missing_start_url",
        )
    queue: list[tuple[str, int]] = [(start, 0)]
    visited: set[str] = set()
    nodes: list[EliNode] = []
    edges: list[EliEdge] = []
    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        metadata_url = _canonical_metadata_url(current)
        try:
            meta = _fetch_json(metadata_url)
        except Exception:
            meta = None
        label = _extract_label(meta or {})
        nodes.append(
            EliNode(
                url=current,
                label=label,
                issue_date=_parse_issue_date(label),
                year_nr=_extract_year_nr(current),
            )
        )
        if not meta or depth >= max_depth:
            continue
        for rel, nxt in _extract_forward_urls(meta):
            edges.append(EliEdge(relation=rel, from_url=current, to_url=nxt))
            if nxt not in visited:
                queue.append((nxt, depth + 1))
    chosen = _best_node(nodes)
    if not chosen:
        return EliResolution(
            start_url=start,
            chosen_url=None,
            chosen_label=None,
            confidence=0.0,
            visited_urls=list(visited),
            edges=edges,
            reason="no_candidates",
        )
    confidence = 0.9 if chosen.url != start else 0.6
    if chosen.issue_date is None:
        confidence -= 0.1
    return EliResolution(
        start_url=start,
        chosen_url=chosen.url,
        chosen_label=chosen.label,
        confidence=max(0.0, min(1.0, confidence)),
        visited_urls=list(visited),
        edges=edges,
        reason=None,
    )
