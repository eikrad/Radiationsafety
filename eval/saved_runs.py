"""Saved evaluation runs: recoverable answers, reports, archives and provenance.

Answers are saved before judging. Finishing writes both reports, then archives
new full-run answers before recording history. Failures keep completed files;
there is no rollback. Dashboard rendering is a separate, subsequent operation.
"""

import gzip
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from langchain_core.documents import Document

from eval.history import append_run, build_run_record, load_runs

_DOCUMENT_FIELDS = ("initial_documents", "documents", "ranked_iaea", "ranked_dk")


@dataclass(frozen=True)
class SavedArtifacts:
    json_report: Path
    markdown_report: Path
    archive: Path | None


class SavedRuns:
    """The local reports and shared history belonging to an evaluation workspace."""

    def __init__(self, output_dir: Path, history_file: Path):
        self.output_dir = output_dir
        self.history_file = history_file

    def claim_run_id(self) -> str:
        """Reserve a UTC timestamp within this output directory, across processes."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        moment = datetime.now(UTC)
        while True:
            run_id = moment.strftime("%Y%m%d_%H%M%S")
            try:
                (self.output_dir / f".run_{run_id}").touch(exist_ok=False)
                return run_id
            except FileExistsError:
                moment += timedelta(seconds=1)

    def save(self, run_id: str, outputs: dict[str, dict]) -> None:
        """Keep answers locally before judging, even if judging later fails."""
        save_outputs(outputs, self._outputs_path(run_id))

    def load(self, run_id: str) -> dict[str, dict] | None:
        """Prefer local answers, then the archive; corrupt files fail visibly.

        Derived run IDs are not followed to their original answers.
        """
        for path in (self._outputs_path(run_id), self._archive_path(run_id)):
            if path.exists():
                return load_outputs(path)
        print(
            f"There are no saved outputs for run {run_id} in {self.output_dir} "
            f"or {self._archive_path(run_id).parent}",
            file=sys.stderr,
        )
        return None

    def provenance(self, run_id: str) -> dict:
        """Original metadata: first history entry, then local report, else unknown."""
        original = next(
            (r for r in load_runs(self.history_file) if r["run_id"] == run_id), None
        )
        if original is not None:
            return original
        report = self._report_paths(run_id)[0]
        return json.loads(report.read_text("utf-8")) if report.exists() else {}

    def judged_passes(self, run_id: str) -> dict[str, bool] | None:
        """Known judge verdicts from the latest history entry; no report fallback."""
        run = next(
            (
                r
                for r in reversed(load_runs(self.history_file))
                if r["run_id"] == run_id
            ),
            None,
        )
        if run is None:
            print(f"Run {run_id} is not in {self.history_file}", file=sys.stderr)
            return None
        return {r["id"]: r["pass"] for r in run["results"] if r.get("pass") is not None}

    def finish(
        self, report: dict, markdown: str, *, record_history: bool = True
    ) -> SavedArtifacts:
        """Save reports, any required answer archive, and finally history.

        Only newly answered full runs require an archive. Retrieval-only runs
        are cheap to repeat; rescores and regrades reuse an existing run's
        answers. Debugging without history creates only local reports.
        """
        run_id = report["run_id"]
        json_path, md_path = self._report_paths(run_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
        md_path.write_text(markdown, "utf-8")
        archive = None
        if record_history:
            record = build_run_record(**report, report_file=json_path.name)
            if not (
                report.get("rescored_from")
                or report.get("regraded_from")
                or report["config"].get("retrieval_only")
            ):
                archive = self._archive_path(run_id)
                archive.parent.mkdir(parents=True, exist_ok=True)
                raw = self._outputs_path(run_id).read_bytes()
                # Stable bytes for the same answers, suitable for version control.
                archive.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
            append_run(record, self.history_file)
        return SavedArtifacts(json_path, md_path, archive)

    def _outputs_path(self, run_id: str) -> Path:
        return self.output_dir / f"outputs_{run_id}.json"

    def _archive_path(self, run_id: str) -> Path:
        return self.history_file.parent / "outputs" / f"{run_id}.json.gz"

    def _report_paths(self, run_id: str) -> tuple[Path, Path]:
        return (
            self.output_dir / f"report_{run_id}.json",
            self.output_dir / f"report_{run_id}.md",
        )


def save_outputs(outputs: dict[str, dict], path: Path) -> None:
    """Write standalone answer records as JSON, including document metadata."""
    serialisable = {
        item_id: {
            **run,
            **{
                field: [
                    {"page_content": d.page_content, "metadata": dict(d.metadata or {})}
                    for d in run.get(field) or []
                ]
                for field in _DOCUMENT_FIELDS
            },
        }
        for item_id, run in outputs.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialisable, ensure_ascii=False), encoding="utf-8")


def load_outputs(path: Path) -> dict[str, dict]:
    """Read standalone JSON or gzip answers, reconstructing their documents."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return {
        item_id: {
            **run,
            **{
                field: [
                    Document(
                        page_content=d.get("page_content", ""),
                        metadata=d.get("metadata", {}),
                    )
                    for d in run.get(field) or []
                ]
                for field in _DOCUMENT_FIELDS
            },
        }
        for item_id, run in data.items()
    }
