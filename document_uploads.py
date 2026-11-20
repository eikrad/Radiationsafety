"""Save, index and register an uploaded PDF as one operation.

Registration requires a completed indexing with at least one chunk. A failed
upload keeps its saved file and any chunks already written; this operation
does not roll back indexing or an overwritten file.
"""

from dataclasses import dataclass
from pathlib import Path

import ingestion
from build_document_sources import (
    _extract_iaea_search_terms,
    _extract_pdf_title_and_version,
    _slug,
)
from document_updates import (
    _lookup_iaea_publication_url_multi,
    append_source_to_registry,
)
from graph import search_index


class NoIndexableContent(ValueError):
    """The PDF yielded no chunks, so it was not registered."""


@dataclass(frozen=True)
class UploadedDocument:
    chunks_added: int
    url: str | None
    problems: tuple[tuple[str, str], ...] = ()

    @property
    def message(self) -> str:
        message = f"Added PDF to collection ({self.chunks_added} chunks)."
        message += "".join(f" Warning: {what}." for _name, what in self.problems)
        if self.url:
            return message + f" URL: {self.url}"
        return message + (
            " No publication URL found (try Build list from local PDFs to refresh)."
        )


def add_uploaded_pdf(
    content: bytes,
    filename: str,
    *,
    folder: str,
    documents_dir: Path,
) -> UploadedDocument:
    """Index an uploaded IAEA PDF, then register it and reopen the search index.

    Raises NoIndexableContent for zero chunks; other failures propagate.
    Registration failures leave the successfully indexed chunks in place.
    """
    if folder not in {"IAEA", "IAEA_other"}:
        raise ValueError("Uploaded PDFs must go to IAEA or IAEA_other")
    safe_name = (
        "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
        or "uploaded.pdf"
    )
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    dest = documents_dir / folder / safe_name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    except OSError as error:
        raise OSError(f"Could not save file: {error}") from error

    title, version = _extract_pdf_title_and_version(dest)
    display_name = (title or safe_name).strip() or safe_name
    url = _lookup_iaea_publication_url_multi(_extract_iaea_search_terms(dest))
    source_id = (
        f"iaea-{_slug(display_name)}"
        if folder == "IAEA"
        else f"iaea-other-{_slug(display_name)}"
    )
    with ingestion.collecting_pdf_problems() as problems:
        chunks = ingestion.add_single_pdf_to_collection(
            dest, folder=folder, source_label=display_name
        )
    if chunks == 0:
        raise NoIndexableContent(
            "No content could be indexed from this PDF; the document was not registered."
        )

    append_source_to_registry(
        source_id=source_id,
        name=display_name,
        url=url,
        folder=folder,
        filename_hint=safe_name,
        version=version,
    )
    search_index.refresh()
    return UploadedDocument(chunks, url or None, tuple(problems))
