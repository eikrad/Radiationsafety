"""Tests for Retsinformation harvest and ELI services."""

from datetime import UTC, datetime

from graph.services import retsinformation_eli as eli
from graph.services import retsinformation_harvest as harvest


def test_normalize_harvest_items():
    """Harvest items are normalized to stable event shape."""
    items = [
        {
            "documentId": "abc",
            "eli": "https://www.retsinformation.dk/eli/lta/2025/1385",
        },
        {"id": "x", "url": "https://www.retsinformation.dk/eli/lta/2025/1384"},
    ]
    out = harvest._normalize_harvest_items(items)
    assert len(out) == 2
    assert out[0].identifier == "abc"
    assert out[1].url.endswith("/1384")


def test_run_incremental_harvest_clamps_to_10_days(tmp_path, monkeypatch):
    """Incremental run clamps start date to provider lookback window."""
    state_path = tmp_path / "sync_state.json"
    state_path.write_text(
        '{"last_successful_date":"2025-01-01","last_run_at":null,"last_error":null}',
        encoding="utf-8",
    )
    calls = []

    def fake_fetch(run_date, api_base=None, subscription_key=None):
        calls.append(run_date.isoformat())
        return []

    monkeypatch.setattr(harvest, "fetch_harvest_documents_for_date", fake_fetch)
    monkeypatch.setattr(harvest, "fetch_eli_update_feed_entries", lambda: [])
    report = harvest.run_incremental_harvest(
        state_path=state_path,
        now=datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
    )
    assert report["start_date"] == "2026-04-05"
    assert calls[0] == "2026-04-05"


def test_resolve_latest_document_prefers_forward_relations(monkeypatch):
    """ELI resolver follows changed_by links and returns latest node."""
    payloads = {
        "https://www.retsinformation.dk/eli/lta/2019/670.json": {
            "title": "BEK nr 670 af 01/01/2019",
            "changed_by": "https://www.retsinformation.dk/eli/lta/2025/1385",
        },
        "https://www.retsinformation.dk/eli/lta/2025/1385.json": {
            "title": "BEK nr 1385 af 18/11/2025"
        },
    }
    monkeypatch.setattr(eli, "_fetch_json", lambda url: payloads.get(url))
    result = eli.resolve_latest_document(
        "https://www.retsinformation.dk/eli/lta/2019/670"
    )
    assert result.chosen_url == "https://www.retsinformation.dk/eli/lta/2025/1385"
    assert result.confidence >= 0.8
