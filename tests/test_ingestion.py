"""Ingestion pipeline tests."""

from pathlib import Path

import pytest

import ingestion
from ingestion import (
    IAEA_COLLECTION,
    DK_LAW_COLLECTION,
    load_dk_law_docs,
    load_iaea_docs,
)


def test_load_iaea_docs_returns_empty_when_no_dirs(tmp_path, monkeypatch):
    """load_iaea_docs returns [] when IAEA dirs do not exist."""
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_iaea_docs()
    assert docs == []


def test_load_iaea_docs_returns_empty_for_empty_iaea_dir(tmp_path, monkeypatch):
    """load_iaea_docs returns [] when IAEA dir exists but has no PDFs."""
    iaea = tmp_path / "IAEA"
    iaea.mkdir()
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_iaea_docs()
    assert docs == []


def test_load_dk_law_docs_returns_empty_when_no_dir(tmp_path, monkeypatch):
    """load_dk_law_docs returns [] when Bekendtgørelse does not exist."""
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_dk_law_docs()
    assert docs == []


def test_load_dk_law_docs_returns_empty_for_empty_dir(tmp_path, monkeypatch):
    """load_dk_law_docs returns [] when Bekendtgørelse exists but has no PDFs."""
    dk = tmp_path / "Bekendtgørelse"
    dk.mkdir()
    monkeypatch.setattr(ingestion, "DOCS_DIR", tmp_path)
    docs = load_dk_law_docs()
    assert docs == []


def test_collection_names():
    """Collection names match expected constants."""
    assert IAEA_COLLECTION == "radiation-iaea"
    assert DK_LAW_COLLECTION == "radiation-dk-law"
