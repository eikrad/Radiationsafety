"""Document source identity and fetched revisions, independent of the search index.

All readers/writers use the same file lock. A durable pending write is replayed
before the next operation, so interrupted YAML/JSON updates converge together.
Legacy version values are retained; they are not evidence of successful indexing.
"""

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


class CatalogError(ValueError):
    """Stored catalog data is invalid; do not replace it with empty state."""


class CatalogChanged(CatalogError):
    """Discovery used an old snapshot; rebuild again instead of losing edits."""


@dataclass(frozen=True)
class DocumentSource:
    id: str
    name: str
    url: str
    folder: str
    filename_hint: str | None
    version: str | None = None


@dataclass(frozen=True)
class CatalogSnapshot:
    entries: list[dict[str, Any]]
    versions: dict[str, dict[str, Any]]
    revision: str

    @property
    def sources(self) -> list[DocumentSource]:
        return [
            DocumentSource(
                id=s["id"],
                name=s["name"],
                url=(s.get("url") or "").strip(),
                folder=s.get("folder", "IAEA"),
                filename_hint=s.get("filename_hint"),
                version=(s.get("version") or "").strip() or None,
            )
            for s in self.entries
            if s.get("id") and s.get("name")
        ]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            if path.exists():
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SourceCatalog:
    """One catalog per registry path; processes must share its companion files.

    Mutations preserve unrelated fields. ``fetched_version`` may only be supplied
    after obtaining the document contents, never after discovering a URL. It does
    not assert that any embedding collection contains that revision.

    Once the pending write is durable, a failed operation may still be completed
    by the next catalog access. Rebuilds require the snapshot's revision to avoid
    replacing concurrent edits. External editors must not write during operations.
    """

    def __init__(
        self,
        registry_path: Path = PROJECT_ROOT / "document_sources.yaml",
        *,
        versions_path: Path | None = None,
        example_path: Path | None = None,
    ):
        self.registry_path = registry_path.resolve()
        self.versions_path = (
            versions_path
            or registry_path.with_name(
                "document_versions.json"
                if registry_path.name == "document_sources.yaml"
                else f"{registry_path.name}.versions.json"
            )
        ).resolve()
        self.example_path = (
            example_path or registry_path.with_name("document_sources.example.yaml")
        ).resolve()
        self._lock_path = self.registry_path.with_name(
            self.registry_path.name + ".lock"
        )
        self._pending_path = self.registry_path.with_name(
            self.registry_path.name + ".pending.json"
        )

    @contextmanager
    def _locked(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._recover()
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _recover(self) -> None:
        if not self._pending_path.exists():
            return
        try:
            pending = json.loads(self._pending_path.read_text(encoding="utf-8"))
            if (
                not isinstance(pending, dict)
                or pending.get("versions_path") != str(self.versions_path)
                or not isinstance(pending.get("registry"), str)
                or not isinstance(pending.get("versions"), str)
            ):
                raise ValueError("invalid pending write")
        except (ValueError, TypeError) as error:
            raise CatalogError(
                f"Cannot recover {self._pending_path}: {error}"
            ) from error
        _atomic_write(self.registry_path, pending["registry"])
        _atomic_write(self.versions_path, pending["versions"])
        self._pending_path.unlink()
        _sync_directory(self._pending_path.parent)

    def _read(self, *, fallback: bool = True) -> tuple[dict, dict]:
        path = self.registry_path
        if fallback and not path.exists():
            path = self.example_path
        try:
            registry = (
                yaml.safe_load(path.read_text(encoding="utf-8"))
                if path.exists()
                else {}
            )
            registry = {} if registry is None else registry
            if not isinstance(registry, dict):
                raise ValueError("registry must be a mapping")
            entries = registry.get("sources", [])
            entries = [] if entries is None else entries
            if not isinstance(entries, list) or not all(
                isinstance(s, dict) for s in entries
            ):
                raise ValueError("sources must be a list of mappings")
            versions = (
                json.loads(self.versions_path.read_text(encoding="utf-8"))
                if self.versions_path.exists()
                else {}
            )
            if not isinstance(versions, dict) or not all(
                isinstance(v, dict) for v in versions.values()
            ):
                raise ValueError("versions must map source IDs to metadata")
        except (ValueError, yaml.YAMLError) as error:
            raise CatalogError(f"Invalid source catalog: {error}") from error
        return registry, versions

    @staticmethod
    def _revision(registry: dict, versions: dict) -> str:
        return hashlib.sha256(
            json.dumps([registry, versions], sort_keys=True).encode()
        ).hexdigest()

    def _commit(self, registry: dict, versions: dict) -> None:
        pending = {
            "versions_path": str(self.versions_path),
            "registry": yaml.safe_dump(registry, allow_unicode=True, sort_keys=False),
            "versions": json.dumps(versions, indent=2),
        }
        _atomic_write(self._pending_path, json.dumps(pending))
        self._recover()

    def snapshot(self) -> CatalogSnapshot:
        with self._locked():
            registry, versions = self._read()
            return CatalogSnapshot(
                registry.get("sources") or [],
                versions,
                self._revision(registry, versions),
            )

    def register(
        self,
        source_id: str,
        name: str,
        *,
        url: str | None = None,
        folder: str = "IAEA_other",
        filename_hint: str | None = None,
        version: str | None = None,
    ) -> str:
        """Register obtained content, choosing the first unused ID suffix.

        As before, registration starts an actual registry rather than copying the
        example. Upload callers must finish indexing before registering.
        """
        with self._locked():
            registry, versions = self._read(fallback=False)
            entries = registry.get("sources") or []
            existing = {s.get("id") for s in entries}
            sid, suffix = source_id, 0
            while sid in existing:
                suffix += 1
                sid = f"{source_id}-{suffix}"
            entries.append(
                {
                    "id": sid,
                    "name": name,
                    "url": (url or "").strip() or None,
                    "folder": folder,
                    "filename_hint": filename_hint,
                    "version": version,
                }
            )
            registry["sources"] = entries
            self._commit(registry, versions)
            return sid

    def update(
        self,
        source_id: str,
        *,
        url: str | None = None,
        fetched_version: str | None = None,
        canonical_eli_url: str | None = None,
        accession_id: str | None = None,
        resolver_confidence: float | None = None,
    ) -> None:
        """Patch source location, fetched revision or identity without losing metadata.

        Registry field updates retain the legacy no-op for absent actual sources;
        version/identity metadata can be recorded for a source read from the example.
        """
        with self._locked():
            registry, versions = self._read(fallback=False)
            for source in registry.get("sources") or []:
                if (source.get("id") or "").strip() == source_id.strip():
                    if url is not None:
                        source["url"] = url.strip()
                    if fetched_version is not None:
                        source["version"] = fetched_version.strip()
                    break
            entry = versions.get(source_id, {}).copy()
            now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            if fetched_version is not None:
                entry.update(version=fetched_version.strip(), updated_at=now)
            if canonical_eli_url or accession_id or resolver_confidence is not None:
                if canonical_eli_url:
                    entry["canonical_eli_url"] = canonical_eli_url.strip()
                if accession_id:
                    entry["accession_id"] = accession_id.strip()
                if resolver_confidence is not None:
                    entry["resolver_confidence"] = f"{resolver_confidence:.3f}"
                entry["identity_updated_at"] = now
            if entry:
                versions[source_id] = entry
            # Do not create an empty registry that would hide the example fallback.
            if not self.registry_path.exists():
                _atomic_write(self.versions_path, json.dumps(versions, indent=2))
            else:
                self._commit(registry, versions)

    def replace_sources(
        self, sources: list[dict[str, Any]], *, expected_revision: str
    ) -> None:
        """Replace the discovered source list only if the catalog is unchanged."""
        with self._locked():
            registry, versions = self._read()
            if self._revision(registry, versions) != expected_revision:
                raise CatalogChanged(
                    "Source catalog changed during discovery; rebuild again"
                )
            ids = [s.get("id") for s in sources]
            if any(not sid for sid in ids) or len(ids) != len(set(ids)):
                raise CatalogError("Discovered sources must have unique, nonempty IDs")
            # Rebuild is intentionally replacement; existing-only sources disappear.
            self._commit({"sources": sources}, versions)


catalog = SourceCatalog()
