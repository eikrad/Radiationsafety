"""Tests for document_updates module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import document_updates as du


def test_load_registry_from_example():
    """Registry loads from example YAML when document_sources.yaml is missing."""
    with patch.object(du, "REGISTRY_PATH", Path("/nonexistent")):
        with patch.object(
            du,
            "REGISTRY_EXAMPLE",
            Path(__file__).resolve().parent.parent / "document_sources.example.yaml",
        ):
            sources = du._load_registry()
    assert len(sources) >= 1
    assert any(
        s.id == "dk-radioaktivitet"
        or s.id == "iaea-safety-assessment-research-reactors"
        for s in sources
    )


def test_load_registry_raw_returns_dicts():
    """load_registry_raw returns list of dicts with id, name, folder; url optional. _load_registry preserves url (or '' if null)."""
    with patch.object(du, "REGISTRY_PATH", Path("/nonexistent")):
        with patch.object(
            du,
            "REGISTRY_EXAMPLE",
            Path(__file__).resolve().parent.parent / "document_sources.example.yaml",
        ):
            raw = du.load_registry_raw()
            assert isinstance(raw, list)
            if raw:
                assert "id" in raw[0] and "name" in raw[0] and "folder" in raw[0]
                loaded = du._load_registry()
                assert loaded[0].url == (raw[0].get("url") or "").strip()


def test_parse_iaea_superseded():
    """IAEA 'Superseded by' is parsed to title and URL."""
    html = 'Superseded by: <a href="/publications/1234/ssg-20-rev1">Specific Safety Guide - SSG-20 (Rev. 1)</a>'
    title, url = du._parse_iaea_superseded(html)
    assert title == "Specific Safety Guide - SSG-20 (Rev. 1)"
    assert url == "https://www.iaea.org/publications/1234/ssg-20-rev1"


def test_parse_iaea_no_superseded():
    """When no Superseded by, returns None."""
    html = "<h1>Current publication</h1>"
    title, url = du._parse_iaea_superseded(html)
    assert title is None
    assert url is None


def test_lookup_iaea_publication_url():
    """_lookup_iaea_publication_url parses search HTML and returns first publication URL."""
    html = '<a href="https://www.iaea.org/publications/8639/safety-assessment-for-research-reactors">Safety Assessment</a>'
    with patch.object(du, "_fetch_url", return_value=html):
        url = du._lookup_iaea_publication_url("Safety Assessment for Research Reactors")
    assert (
        url
        == "https://www.iaea.org/publications/8639/safety-assessment-for-research-reactors"
    )


def test_lookup_iaea_publication_url_relative_href():
    """_lookup_iaea_publication_url accepts relative /publications/ID/slug links."""
    html = '<a href="/publications/14812/ssg-20-rev-1">SSG-20 Rev 1</a>'
    with patch.object(du, "_fetch_url", return_value=html):
        url = du._lookup_iaea_publication_url("SSG-20")
    assert url == "https://www.iaea.org/publications/14812/ssg-20-rev-1"


def test_lookup_iaea_publication_url_multi():
    """_lookup_iaea_publication_url_multi tries each query and returns first URL found."""
    html1 = ""
    html2 = (
        '<a href="https://www.iaea.org/publications/8639/safety-assessment">Link</a>'
    )
    with patch.object(du, "_fetch_url", side_effect=[html1, html2]):
        url = du._lookup_iaea_publication_url_multi(
            ["nonexistent", "safety assessment"]
        )
    assert url == "https://www.iaea.org/publications/8639/safety-assessment"


def test_lookup_iaea_publication_url_via_brave_returns_first_publication_url():
    """_lookup_iaea_publication_url_via_brave parses Brave API response and returns first iaea.org/publications URL."""
    body = json.dumps(
        {
            "web": {
                "results": [
                    {
                        "url": "https://www.iaea.org/publications/10677/risk-informed-approach"
                    }
                ]
            },
        }
    ).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=None)
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}):
        with patch("document_updates.urllib.request.urlopen", return_value=mock_resp):
            url = du._lookup_iaea_publication_url_via_brave("STI/PUB/1678")
    assert url == "https://www.iaea.org/publications/10677/risk-informed-approach"


def test_lookup_iaea_publication_url_via_brave_no_key_returns_none():
    """_lookup_iaea_publication_url_via_brave returns None when BRAVE_SEARCH_API_KEY is not set."""
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        url = du._lookup_iaea_publication_url_via_brave("SSG-20")
    assert url is None


def test_parse_retsinformation_amendments():
    """Retsinformation amendments list is parsed; newest by date returned."""
    html = """
    <a href="/eli/lta/2024/476">BEK nr 476 af 13/05/2024</a>
    <a href="/eli/lta/2025/1385">BEK nr 1385 af 18/11/2025</a>
    <a href="/eli/lta/2025/645">BEK nr 645 af 06/06/2025</a>
    """
    label, url = du._parse_retsinformation(
        html, "https://www.retsinformation.dk/eli/lta/2019/670"
    )
    assert label == "BEK nr 1385 af 18/11/2025"
    assert "1385" in url
    assert url.startswith("https://www.retsinformation.dk")


def test_parse_retsinformation_fallback_by_nr():
    """When link and BEK text are not adjacent, fallback matches by BEK number."""
    html = """
    <ul><li><a href="/eli/lta/2025/1385">Se dokument</a></li></ul>
    <p>Senere ændringer: BEK nr 1385 af 18/11/2025.</p>
    """
    label, url = du._parse_retsinformation(
        html, "https://www.retsinformation.dk/eli/lta/2019/670"
    )
    assert label == "BEK nr 1385 af 18/11/2025"
    assert "1385" in url


def test_allowed_url():
    """Only allowlisted hosts are allowed."""
    assert du._allowed_url("https://www.retsinformation.dk/eli/lta/2019/670") is True
    assert du._allowed_url("https://www.iaea.org/publications/123") is True
    assert du._allowed_url("https://www.sst.dk/media/foo/bar.pdf") is True
    assert du._allowed_url("https://evil.com/foo") is False


def test_is_sst_source():
    """SST sources are detected by URL or by document name (åbne radioaktive kilder, sikkerhedsvurdering)."""
    assert (
        du._is_sst_source(
            du.DocumentSource(
                id="x",
                name="x",
                url="https://www.sst.dk/media/foo/bar.pdf",
                folder="Bekendtgørelse",
                filename_hint=None,
            )
        )
        is True
    )
    assert (
        du._is_sst_source(
            du.DocumentSource(
                id="x",
                name="Brug af åbne radioaktive kilder",
                url="",
                folder="Bekendtgørelse",
                filename_hint=None,
            )
        )
        is True
    )
    assert (
        du._is_sst_source(
            du.DocumentSource(
                id="x",
                name="Udarbejdelse af en sikkerhedsvurdering",
                url="",
                folder="Bekendtgørelse",
                filename_hint=None,
            )
        )
        is True
    )
    assert (
        du._is_sst_source(
            du.DocumentSource(
                id="x",
                name="Bekendtgørelse om ioniserende stråling",
                url="https://www.retsinformation.dk/eli/lta/2019/670",
                folder="Bekendtgørelse",
                filename_hint=None,
            )
        )
        is False
    )


def test_extract_bek_number():
    """BEK number is extracted from URL or version string."""
    assert (
        du._extract_bek_number(
            du.DocumentSource(
                id="x",
                name="x",
                url="https://www.retsinformation.dk/eli/lta/2019/670",
                folder="Bekendtgørelse",
                filename_hint=None,
            )
        )
        == 670
    )
    assert (
        du._extract_bek_number(
            du.DocumentSource(
                id="x",
                name="x",
                url="",
                folder="Bekendtgørelse",
                filename_hint=None,
                version="BEK nr 1385 af 18/11/2025",
            )
        )
        == 1385
    )
    assert (
        du._extract_bek_number(
            du.DocumentSource(
                id="x", name="x", url="", folder="Bekendtgørelse", filename_hint=None
            )
        )
        is None
    )


def test_version_string_to_date_nr_parses_issue_date():
    """BEK version labels parse to (date, number) for robust recency checks."""
    parsed = du._version_string_to_date_nr("BEK nr 1385 af 18/11/2025")
    assert parsed is not None
    d, nr = parsed
    assert d.isoformat() == "2025-11-18"
    assert nr == 1385


def test_reject_if_older_than_current_uses_issue_date_not_url_number():
    """Reject a remote candidate when issue date is older, even if URL year/number looks newer."""
    source = du.DocumentSource(
        id="dk-radio",
        name="Radioaktivitetsbekendtgørelsen",
        url="https://www.retsinformation.dk/eli/lta/2024/9999",
        folder="Bekendtgørelse",
        filename_hint=None,
    )
    label, url = du._reject_if_older_than_current(
        remote_label="BEK nr 42 af 01/01/2025",
        download_url="https://www.retsinformation.dk/eli/lta/2025/42",
        current_url=source.url,
        current_version="BEK nr 1385 af 18/11/2025",
        source=source,
    )
    assert label is None
    assert url is None


def test_danish_source_gets_url_via_probe():
    """Danish source with BEK number gets URL via probe (no API)."""
    source = du.DocumentSource(
        id="dk-radio",
        name="Radioaktivitetsbekendtgørelsen",
        url="https://www.retsinformation.dk/eli/lta/2019/670",
        folder="Bekendtgørelse",
        filename_hint=None,
    )
    with patch.object(
        du,
        "_resolve_danish_url_by_probing",
        return_value=("BEK nr 670", "https://www.retsinformation.dk/eli/lta/2024/670"),
    ):
        result = du.check_one_source(source, {})
    assert result["download_url"] == "https://www.retsinformation.dk/eli/lta/2024/670"
    assert result["remote_version"] == "BEK nr 670"


def test_danish_search_follows_amendments_to_newest():
    """Search-resolved Danish URL is upgraded via 'Senere ændringer' before return."""
    with patch.object(
        du,
        "_resolve_danish_url_via_brave",
        return_value="https://www.retsinformation.dk/eli/lta/2019/670",
    ):
        with patch.object(
            du,
            "_resolve_danish_url_to_newest",
            return_value=(
                "BEK nr 1385 af 18/11/2025",
                "https://www.retsinformation.dk/eli/lta/2025/1385",
            ),
        ):
            label, url = du._resolve_danish_url_by_search(
                "Bekendtgørelse om ioniserende stråling"
            )
    assert label == "BEK nr 1385 af 18/11/2025"
    assert url == "https://www.retsinformation.dk/eli/lta/2025/1385"


def test_check_one_source_guarded_prefers_eli(monkeypatch):
    """Guarded mode uses ELI resolver when available."""
    source = du.DocumentSource(
        id="dk-radio",
        name="Radioaktivitetsbekendtgørelsen",
        url="https://www.retsinformation.dk/eli/lta/2019/670",
        folder="Bekendtgørelse",
        filename_hint=None,
    )
    monkeypatch.setattr(du, "RETSINFO_RESOLVER_MODE", "guarded")
    with patch.object(
        du, "_resolve_danish_source", return_value=du.ResolvedUrl("legacy", source.url)
    ):
        with patch.object(
            du,
            "_resolve_danish_source_via_eli",
            return_value=(
                du.ResolvedUrl(
                    "BEK nr 1385 af 18/11/2025",
                    "https://www.retsinformation.dk/eli/lta/2025/1385",
                ),
                {"confidence": 0.9},
            ),
        ):
            result = du.check_one_source(source, {})
    assert result["download_url"] == "https://www.retsinformation.dk/eli/lta/2025/1385"
    assert result["resolver_source"] == "eli"


def test_sync_danish_legislation_apply_updates(monkeypatch):
    """Incremental sync updates URL/version when apply_updates=true and URL changed."""
    source = du.DocumentSource(
        id="dk-radio",
        name="Radioaktivitetsbekendtgørelsen",
        url="https://www.retsinformation.dk/eli/lta/2019/670",
        folder="Bekendtgørelse",
        filename_hint=None,
        version="BEK nr 670 af 01/01/2019",
    )
    monkeypatch.setattr(du, "_load_registry", lambda: [source])
    monkeypatch.setattr(du, "_load_versions", lambda: {})
    monkeypatch.setattr(du, "_reset_runtime_caches", lambda: None)
    monkeypatch.setattr(
        du,
        "run_incremental_harvest",
        lambda **kwargs: {"harvest_events": [], "eli_feed_events": [], "errors": []},
    )
    monkeypatch.setattr(
        du,
        "_resolve_danish_source_via_eli",
        lambda *args, **kwargs: (
            du.ResolvedUrl(
                "BEK nr 1385 af 18/11/2025",
                "https://www.retsinformation.dk/eli/lta/2025/1385",
            ),
            {"confidence": 0.92},
        ),
    )
    updated: list[tuple[str, str]] = []
    versions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        du, "update_registry_url", lambda sid, url: updated.append((sid, url))
    )
    monkeypatch.setattr(
        du, "update_version_after_ingest", lambda sid, ver: versions.append((sid, ver))
    )
    monkeypatch.setattr(du, "update_source_identity", lambda *args, **kwargs: None)
    report = du.sync_danish_legislation(apply_updates=True)
    assert report["updated_count"] == 1
    assert updated == [("dk-radio", "https://www.retsinformation.dk/eli/lta/2025/1385")]
    assert versions == [("dk-radio", "BEK nr 1385 af 18/11/2025")]


def test_get_current_version_from_file_bekendtgoerelse(tmp_path):
    """Current version is read from {source_id}_version.txt for Bekendtgørelse."""
    bek = tmp_path / "documents" / "Bekendtgørelse"
    bek.mkdir(parents=True)
    (bek / "dk-radio_version.txt").write_text(
        "BEK nr 1385 af 18/11/2025", encoding="utf-8"
    )
    source = du.DocumentSource(
        id="dk-radio",
        name="Radioaktivitetsbekendtgørelsen",
        url="https://www.retsinformation.dk/eli/lta/2019/670",
        folder="Bekendtgørelse",
        filename_hint=None,
    )
    with patch.object(du, "DOCS_DIR", tmp_path / "documents"):
        version = du._get_current_version_from_file(source)
    assert version == "BEK nr 1385 af 18/11/2025"


def test_update_registry_url(tmp_path):
    """update_registry_url updates the source url in document_sources.yaml."""
    yaml_path = tmp_path / "document_sources.yaml"
    yaml_path.write_text(
        """
sources:
  - id: dk-1
    name: Test
    url: "https://www.retsinformation.dk/eli/lta/2019/670"
    folder: Bekendtgørelse
""",
        encoding="utf-8",
    )
    with patch.object(du, "REGISTRY_PATH", yaml_path):
        du.update_registry_url(
            "dk-1", "https://www.retsinformation.dk/eli/lta/2025/1385"
        )
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    urls = [s["url"] for s in data["sources"] if s.get("id") == "dk-1"]
    assert urls == ["https://www.retsinformation.dk/eli/lta/2025/1385"]


def test_update_registry_version(tmp_path):
    """update_registry_version writes the version into document_sources.yaml."""
    yaml_path = tmp_path / "document_sources.yaml"
    yaml_path.write_text(
        """
sources:
  - id: dk-1
    name: Test
    url: "https://www.retsinformation.dk/eli/lta/2025/1385"
    folder: Bekendtgørelse
""",
        encoding="utf-8",
    )
    with patch.object(du, "REGISTRY_PATH", yaml_path):
        du.update_registry_version("dk-1", "BEK nr 1385 af 18/11/2025")
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    versions = [s.get("version") for s in data["sources"] if s.get("id") == "dk-1"]
    assert versions == ["BEK nr 1385 af 18/11/2025"]
