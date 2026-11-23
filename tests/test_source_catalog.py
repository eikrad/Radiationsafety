"""Catalog contracts through real files, processes, and interrupted writes."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

import source_catalog
from source_catalog import CatalogChanged, CatalogError, SourceCatalog


def test_example_fallback_and_null_url(tmp_path):
    example = tmp_path / "example.yaml"
    example.write_text("sources:\n- id: guide\n  name: Guide\n  url: null\n")
    catalog = SourceCatalog(tmp_path / "sources.yaml", example_path=example)
    snapshot = catalog.snapshot()
    assert snapshot.entries[0]["url"] is None
    assert snapshot.sources[0].url == ""
    assert snapshot.sources[0].folder == "IAEA"
    # Legacy update behavior: example is never rewritten or hidden by empty YAML.
    catalog.update("guide", fetched_version="v1")
    assert not catalog.registry_path.exists()
    assert catalog.snapshot().versions["guide"]["version"] == "v1"
    assert catalog.snapshot().sources[0].id == "guide"
    catalog.register("local", "Local")
    assert [s.id for s in catalog.snapshot().sources] == ["local"]


def test_registration_uses_first_available_suffix(catalog):
    catalog.registry_path.write_text(
        "sources:\n- id: guide\n- id: guide-1\n- id: guide-3\n"
    )
    assert catalog.register("guide", "New guide") == "guide-2"
    assert [s["id"] for s in catalog.snapshot().entries] == [
        "guide",
        "guide-1",
        "guide-3",
        "guide-2",
    ]


def test_fetched_revision_preserves_identity_and_unknown_metadata(catalog):
    catalog.register("law", "Law", version="old")
    catalog.update(
        "law",
        canonical_eli_url=" https://example.org/law ",
        accession_id=" A123 ",
        resolver_confidence=0.9123,
    )
    versions = json.loads(catalog.versions_path.read_text())
    versions["law"]["future_field"] = {"keep": True}
    catalog.versions_path.write_text(json.dumps(versions))
    registry = yaml.safe_load(catalog.registry_path.read_text())
    registry["owner"] = "team"
    registry["sources"][0]["extra"] = "keep"
    catalog.registry_path.write_text(yaml.safe_dump(registry))

    catalog.update("law", url=" https://example.org/new ", fetched_version=" new ")

    snapshot = catalog.snapshot()
    assert snapshot.sources[0].version == "new"
    assert snapshot.sources[0].url == "https://example.org/new"
    assert snapshot.versions["law"] == {
        **versions["law"],
        "version": "new",
        "updated_at": snapshot.versions["law"]["updated_at"],
    }
    assert snapshot.entries[0]["extra"] == "keep"
    assert yaml.safe_load(catalog.registry_path.read_text())["owner"] == "team"
    catalog.update("law", url="https://example.org/discovered")
    assert catalog.snapshot().versions == snapshot.versions
    assert catalog.snapshot().sources[0].version == "new"


@pytest.mark.parametrize(
    "target,content",
    [
        ("registry_path", "sources: [bad]"),
        ("registry_path", "[bad]"),
        ("registry_path", "sources: false"),
        ("versions_path", "{broken"),
        ("versions_path", "[]"),
        ("versions_path", '{"law": "bad"}'),
    ],
)
def test_invalid_persisted_state_is_not_overwritten(catalog, target, content):
    catalog.register("law", "Law")
    path = getattr(catalog, target)
    path.write_text(content)
    with pytest.raises(CatalogError):
        catalog.update("law", fetched_version="new")
    assert path.read_text() == content
    assert not catalog._pending_path.exists()


@pytest.mark.parametrize("failure_at", ["journal", "registry", "versions"])
def test_interrupted_commit_recovers_before_another_reader(
    catalog, monkeypatch, failure_at
):
    catalog.register("law", "Law", version="old")
    catalog.update("law", fetched_version="old", accession_id="A123")
    target = {
        "journal": catalog._pending_path,
        "registry": catalog.registry_path,
        "versions": catalog.versions_path,
    }[failure_at]
    write = source_catalog._atomic_write

    def interrupted(path, content):
        if path == target:
            raise OSError("interrupted")
        write(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(source_catalog, "_atomic_write", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            catalog.update("law", url="https://example.org/new", fetched_version="new")
    # A fresh instance/process recovers a durable pending write before returning.
    recovered = SourceCatalog(catalog.registry_path).snapshot()
    expected = "old" if failure_at == "journal" else "new"
    assert (
        recovered.sources[0].version == recovered.versions["law"]["version"] == expected
    )
    assert recovered.versions["law"]["accession_id"] == "A123"
    assert not catalog._pending_path.exists()


def test_failed_atomic_replace_leaves_the_previous_file_complete(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_text("old")

    def fail(*args):
        raise OSError("disk error")

    monkeypatch.setattr(source_catalog.os, "replace", fail)
    with pytest.raises(OSError):
        source_catalog._atomic_write(path, "new")
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def _register_in_process(path, count):
    catalog = SourceCatalog(Path(path))
    for _ in range(count):
        catalog.register("guide", "Guide")


@pytest.mark.parametrize("executor", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_concurrent_registrations_do_not_lose_sources(catalog, executor):
    options = (
        {"mp_context": multiprocessing.get_context("spawn")}
        if executor is ProcessPoolExecutor
        else {}
    )
    with executor(max_workers=4, **options) as pool:
        futures = [
            pool.submit(_register_in_process, str(catalog.registry_path), 5)
            for _ in range(4)
        ]
        for future in futures:
            future.result(timeout=20)
    ids = {s.id for s in catalog.snapshot().sources}
    assert ids == {"guide", *(f"guide-{n}" for n in range(1, 20))}


def test_concurrent_identity_and_revision_updates_are_merged(catalog):
    catalog.register("law", "Law")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                SourceCatalog(catalog.registry_path).update,
                "law",
                fetched_version="new",
            ),
            pool.submit(
                SourceCatalog(catalog.registry_path).update, "law", accession_id="A123"
            ),
            pool.submit(
                SourceCatalog(catalog.registry_path).update,
                "law",
                canonical_eli_url="https://example.org/law",
            ),
        ]
        for future in futures:
            future.result(timeout=10)
    entry = catalog.snapshot().versions["law"]
    assert entry["version"] == "new"
    assert entry["accession_id"] == "A123"
    assert entry["canonical_eli_url"] == "https://example.org/law"


def test_rebuild_rejects_stale_discovery_and_retains_replacement_semantics(catalog):
    catalog.register("old", "Old")
    before = catalog.snapshot()
    catalog.register("uploaded", "Uploaded")
    discovered = [{"id": "new", "name": "New", "version": "v1"}]
    with pytest.raises(CatalogChanged, match="rebuild again"):
        catalog.replace_sources(discovered, expected_revision=before.revision)
    assert {s.id for s in catalog.snapshot().sources} == {"old", "uploaded"}
    catalog.replace_sources(discovered, expected_revision=catalog.snapshot().revision)
    assert catalog.snapshot().entries == discovered


def test_rebuild_rejects_duplicate_source_ids(catalog):
    with pytest.raises(CatalogError, match="unique"):
        catalog.replace_sources(
            [{"id": "same"}, {"id": "same"}],
            expected_revision=catalog.snapshot().revision,
        )
    assert catalog.snapshot().entries == []
