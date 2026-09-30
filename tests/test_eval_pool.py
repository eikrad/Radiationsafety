"""Pooling: retrieved chunks that no evidence quote covers, for human review."""

import json

from langchain_core.documents import Document

from eval.graph_run import save_outputs
from eval.pool import main, pool_candidates

ITEM = {
    "id": "dk-dose-limits",
    "question": "Hvor findes dosisgrænserne?",
    "expected_behavior": "answer",
    "nuggets": [
        {"text": "Annex 2", "importance": "vital", "evidence": ["fremgår af bilag 2"]},
    ],
}
EVIDENCE = Document(
    page_content="Grænserne fremgår af bilag 2.", metadata={"source": "BEK 1"}
)
OTHER = Document(
    page_content="Dosisgrænser: se bilag 2 til bekendtgørelsen.",
    metadata={"source": "BEK 2"},
)
NOISE = Document(page_content="Emballage til transport.", metadata={"source": "SSR-6"})


def test_chunks_in_the_top_k_without_evidence_are_candidates():
    runs = {"run-a": {"iaea": [NOISE], "dk": [OTHER, EVIDENCE]}}

    candidates = pool_candidates(ITEM, runs, k=2)

    assert [c["text"] for c in candidates] == [NOISE.page_content, OTHER.page_content]
    assert candidates[1] == {
        "text": OTHER.page_content,
        "source": "BEK 2",
        "collection": "dk",
        "found_by": {"run-a": 1},
    }


def test_chunks_below_the_top_k_are_not_candidates():
    runs = {"run-a": {"iaea": [], "dk": [EVIDENCE, OTHER]}}

    assert pool_candidates(ITEM, runs, k=1) == []


def test_a_chunk_found_by_several_runs_is_listed_once_with_its_ranks():
    runs = {
        "dense": {"iaea": [], "dk": [EVIDENCE, OTHER]},
        "hybrid": {"iaea": [], "dk": [OTHER, EVIDENCE]},
    }

    [candidate] = pool_candidates(ITEM, runs, k=2)

    assert candidate["found_by"] == {"dense": 2, "hybrid": 1}


def test_the_pool_report_lists_questions_where_a_run_missed_the_evidence(
    tmp_path, monkeypatch, capsys
):
    golden = tmp_path / "golden.json"
    found = {**ITEM, "id": "found", "question": "Found?"}
    golden.write_text(json.dumps([ITEM, found]), encoding="utf-8")
    save_outputs(
        {
            "dk-dose-limits": {"ranked_iaea": [NOISE], "ranked_dk": [OTHER]},
            "found": {"ranked_iaea": [NOISE], "ranked_dk": [EVIDENCE]},
        },
        tmp_path / "outputs_run-a.json",
    )
    report = tmp_path / "pool.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pool",
            "run-a",
            "--golden",
            str(golden),
            "--reports-dir",
            str(tmp_path),
            "--k",
            "3",
            "--output",
            str(report),
        ],
    )

    assert main() == 0

    text = report.read_text(encoding="utf-8")
    assert "dk-dose-limits" in text and OTHER.page_content in text
    assert "## found" not in text  # every run found its evidence: nothing to review
