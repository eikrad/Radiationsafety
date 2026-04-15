"""Retsinformation Harvest + ELI feed utilities."""

from __future__ import annotations

import json
import ssl
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

HARVEST_API_BASE = "https://api.retsinformation.dk"
HARVEST_MIN_SECONDS_BETWEEN_CALLS = 10.0
ELI_UPDATE_FEED_URL = "https://www.retsinformation.dk/eli/eli-update-feed.atom"

_harvest_lock = threading.Lock()
_last_harvest_request: list[float] = [0.0]


@dataclass
class HarvestSyncState:
    """Persistent checkpoint for incremental harvest sync."""

    last_successful_date: str | None = None
    last_run_at: str | None = None
    last_error: str | None = None


@dataclass
class HarvestDocumentEvent:
    """One normalized changed-document event from harvest API or feed."""

    source: str
    identifier: str
    url: str
    changed_at: str | None = None
    reason: str | None = None


def _throttle_harvest_requests() -> None:
    """Enforce provider limit of one request per 10 seconds."""
    with _harvest_lock:
        now = time.monotonic()
        elapsed = now - _last_harvest_request[0]
        if elapsed < HARVEST_MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(HARVEST_MIN_SECONDS_BETWEEN_CALLS - elapsed)
        _last_harvest_request[0] = time.monotonic()


def _json_get(url: str, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "RadiationSafetyRAG/1.0",
            **(headers or {}),
        },
    )
    _throttle_harvest_requests()
    with urllib.request.urlopen(
        req, timeout=30, context=ssl.create_default_context()
    ) as resp:
        return json.load(resp)


def load_harvest_state(path: Path) -> HarvestSyncState:
    """Load persisted harvest sync state."""
    if not path.exists():
        return HarvestSyncState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HarvestSyncState()
    return HarvestSyncState(
        last_successful_date=raw.get("last_successful_date"),
        last_run_at=raw.get("last_run_at"),
        last_error=raw.get("last_error"),
    )


def save_harvest_state(path: Path, state: HarvestSyncState) -> None:
    """Persist harvest checkpoint state."""
    payload = {
        "last_successful_date": state.last_successful_date,
        "last_run_at": state.last_run_at,
        "last_error": state.last_error,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_harvest_items(items: Any) -> list[HarvestDocumentEvent]:
    out: list[HarvestDocumentEvent] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        identifier = str(
            item.get("documentId")
            or item.get("id")
            or item.get("accessionNumber")
            or item.get("accessionsnummer")
            or ""
        ).strip()
        url = str(
            item.get("eli")
            or item.get("url")
            or item.get("href")
            or item.get("documentUrl")
            or ""
        ).strip()
        if not identifier and not url:
            continue
        out.append(
            HarvestDocumentEvent(
                source="harvest_api",
                identifier=identifier or url,
                url=url,
                changed_at=str(
                    item.get("changedAt")
                    or item.get("updatedAt")
                    or item.get("modifiedAt")
                    or ""
                )
                or None,
                reason=str(item.get("reasonForChange") or item.get("reason") or "")
                or None,
            )
        )
    return out


def fetch_harvest_documents_for_date(
    run_date: date,
    api_base: str = HARVEST_API_BASE,
    subscription_key: str | None = None,
) -> list[HarvestDocumentEvent]:
    """Fetch changed documents for one date via /v1/Documents."""
    params = urllib.parse.urlencode({"date": run_date.isoformat()})
    url = f"{api_base.rstrip('/')}/v1/Documents?{params}"
    headers: dict[str, str] = {}
    if subscription_key:
        headers["Ocp-Apim-Subscription-Key"] = subscription_key.strip()
    payload = _json_get(url, headers=headers)
    if isinstance(payload, dict):
        items = (
            payload.get("documents")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    else:
        items = payload
    return _normalize_harvest_items(items)


def fetch_eli_update_feed_entries(
    feed_url: str = ELI_UPDATE_FEED_URL,
) -> list[HarvestDocumentEvent]:
    """Fetch ELI atom feed entries as fallback/reconciliation events."""
    req = urllib.request.Request(
        feed_url,
        headers={
            "Accept": "application/atom+xml",
            "User-Agent": "RadiationSafetyRAG/1.0",
        },
    )
    with urllib.request.urlopen(
        req, timeout=30, context=ssl.create_default_context()
    ) as resp:
        content = resp.read()
    root = ET.fromstring(content)
    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    out: list[HarvestDocumentEvent] = []
    for entry in root.findall("atom:entry", atom_ns):
        entry_id = (
            entry.findtext("atom:id", default="", namespaces=atom_ns) or ""
        ).strip()
        updated = (
            entry.findtext("atom:updated", default="", namespaces=atom_ns) or ""
        ).strip()
        link_el = entry.find("atom:link", atom_ns)
        href = ""
        if link_el is not None:
            href = (link_el.attrib.get("href") or "").strip()
        url = href or entry_id
        if not url:
            continue
        out.append(
            HarvestDocumentEvent(
                source="eli_atom_feed",
                identifier=entry_id or url,
                url=url,
                changed_at=updated or None,
                reason="eli_feed_update",
            )
        )
    return out


def run_incremental_harvest(
    *,
    state_path: Path,
    api_base: str = HARVEST_API_BASE,
    subscription_key: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one incremental harvest sync and persist checkpoint state."""
    run_now = now or datetime.now(UTC)
    state = load_harvest_state(state_path)
    start_date: date
    if state.last_successful_date:
        try:
            start_date = date.fromisoformat(state.last_successful_date) + timedelta(
                days=1
            )
        except ValueError:
            start_date = run_now.date() - timedelta(days=1)
    else:
        start_date = run_now.date() - timedelta(days=1)
    # API supports last ~10 days; clamp window.
    min_supported = run_now.date() - timedelta(days=10)
    if start_date < min_supported:
        start_date = min_supported
    end_date = run_now.date()
    harvest_events: list[HarvestDocumentEvent] = []
    errors: list[str] = []
    d = start_date
    while d <= end_date:
        try:
            harvest_events.extend(
                fetch_harvest_documents_for_date(
                    d, api_base=api_base, subscription_key=subscription_key
                )
            )
        except Exception as exc:  # pragma: no cover - network behavior
            errors.append(f"{d.isoformat()}: {exc}")
        d += timedelta(days=1)
    feed_events: list[HarvestDocumentEvent] = []
    try:
        feed_events = fetch_eli_update_feed_entries()
    except Exception as exc:  # pragma: no cover - network behavior
        errors.append(f"eli-feed: {exc}")
    state.last_run_at = run_now.isoformat()
    if errors:
        state.last_error = "; ".join(errors)[:1000]
    else:
        state.last_error = None
        state.last_successful_date = end_date.isoformat()
    save_harvest_state(state_path, state)
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "harvest_events": [e.__dict__ for e in harvest_events],
        "eli_feed_events": [e.__dict__ for e in feed_events],
        "errors": errors,
        "state": state.__dict__,
    }
