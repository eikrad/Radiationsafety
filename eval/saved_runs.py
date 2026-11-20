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
from typing import Never

from langchain_core.documents import Document

from eval.history import append_run, build_run_record, load_runs

_DOCUMENT_FIELDS = ("initial_documents", "documents", "ranked_iaea", "ranked_dk")


class SavedRunProvenanceError(ValueError):
    """Saved answers cannot be attributed to the requested original questions."""


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

    def rescore_provenance(self, run_id: str, questions: list[dict]) -> dict:
        """Require original settings and exact text for the questions being reused.

        Questions may be reordered or omitted, and their grading criteria may
        change. An ID alone does not establish that the saved answer belongs to
        today's question. Unknown metadata must never be replaced by today's
        model configuration or git state.
        """

        return self._validated_provenance(run_id, questions, operation="rescore")

    def _validated_provenance(
        self, run_id: str, questions: list[dict], *, operation: str
    ) -> dict:
        def reject(reason: str) -> Never:
            raise SavedRunProvenanceError(
                f"Cannot {operation} {run_id}: {reason}. Restore the original "
                "history/report and question text, or run a new evaluation."
            )

        try:
            original = self.provenance(run_id)
        except json.JSONDecodeError:
            reject("original report is not valid JSON")
        if not isinstance(original, dict) or not original:
            reject("missing original metadata")
        if original.get("run_id") != run_id:
            reject("original metadata identifies a different run")

        config = original.get("config")
        if not isinstance(config, dict) or not config:
            reject("missing original configuration")
        git = original.get("git")
        if not isinstance(git, dict) or not _known_text(git.get("commit")):
            reject("missing original git commit")
        if "retrieval_only" in config and not isinstance(
            config["retrieval_only"], bool
        ):
            reject("invalid original retrieval_only setting")
        if not config.get("retrieval_only"):
            for key in ("llm_provider", "llm_model"):
                if not _known_text(config.get(key)):
                    reject(f"missing original {key}")

        if not questions:
            reject("no saved questions match the selected golden set")
        problems = _question_problems(original.get("results"), questions)
        if problems:
            reject("; ".join(problems))
        return original

    def regrade_labels(
        self, run_id: str, questions: list[dict], *, labels_from: str | None = None
    ) -> dict[str, bool]:
        """Validate saved questions and select judge labels for those same answers.

        Labels must be recorded in history under the original run or a direct
        rescore of it. Keep the latest label record, allowing partial rescores
        and unjudged questions; those contribute no answer-grader accuracy.
        """
        self._validated_provenance(run_id, questions, operation="regrade")
        label_id = labels_from or run_id

        def reject(reason: str) -> Never:
            raise SavedRunProvenanceError(
                f"Cannot regrade {run_id} with labels from {label_id}: {reason}. "
                "Choose the original run or a recorded rescore of its answers "
                "with matching question text."
            )

        run = next(
            (
                r
                for r in reversed(load_runs(self.history_file))
                if r["run_id"] == label_id
            ),
            None,
        )
        if run is None:
            reject("label run is not recorded in history")
        if label_id != run_id and run.get("rescored_from") != run_id:
            reject(f"label run is not a rescore of {run_id}")
        config = run.get("config")
        if not isinstance(config, dict):
            reject("missing label run configuration")
        if run.get("regraded_from") or config.get("grader_only"):
            reject("a regrade does not supply judge labels")
        problems = _question_problems(run.get("results"), questions, allow_missing=True)
        if problems:
            reject("; ".join(problems))
        selected = {q["id"] for q in questions}
        labels = {}
        for result in run["results"]:
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("id"), str)
                or result["id"] not in selected
            ):
                continue
            verdict = result.get("pass")
            if verdict is None:
                continue
            if not isinstance(verdict, bool):
                reject(f"{result['id']}: judge label must be true, false, or null")
            labels[result["id"]] = verdict
        return labels

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


def _question_problems(
    results: object, questions: list[dict], *, allow_missing: bool = False
) -> list[str]:
    if not isinstance(results, list):
        return ["missing original questions"]
    problems = []
    for question in questions:
        item_id = question["id"]
        matches = [r for r in results if isinstance(r, dict) and r.get("id") == item_id]
        if not matches and allow_missing:
            continue
        if len(matches) != 1 or not _known_text(matches[0].get("question")):
            problems.append(f"{item_id}: missing or ambiguous original question")
        elif matches[0]["question"] != question["question"]:
            problems.append(f"{item_id}: question text changed")
    return problems


def _known_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
