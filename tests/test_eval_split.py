"""Splitting the held-out set into tune and confirm, one index chunk at a time."""

import json
from collections import Counter

import pytest

import eval.golden as golden_cli
import eval.split as split_cli
from eval.split import assign_splits, passage_groups, select_split, split_leaks


def _item(item_id, *quotes, **tags):
    item = {
        "id": item_id,
        "question": f"Question {item_id}?",
        "topics": ["general"],
        "language": "da",
        "source": "dk-law",
        "expected_behavior": "answer",
        "difficulty": "medium",
        "nuggets": [
            {"text": f"fact {n}", "importance": "vital", "evidence": [quote]}
            for n, quote in enumerate(quotes, start=1)
        ],
    }
    item.update(tags)
    return item


SECTION_75 = (
    "§ 75. Stråleudsatte arbejdstagere i kategori A og kategori B skal være omfattet "
    "af individuel dosisovervågning. Stk. 4. Effektiv dosis skal bestemmes med "
    "persondosimeter med en måleperiode på højst 3 måneder."
)
SECTION_91 = "§ 91. Referenceniveauet for radon i bygninger er 100 Bq/m^3."


def test_items_whose_evidence_is_in_the_same_chunk_form_one_group():
    category = _item("category", "i kategori A og kategori B skal være omfattet")
    dosimeter = _item("dosimeter", "måleperiode på højst 3 måneder", language="en")
    radon = _item("radon", "radon i bygninger er 100 Bq/m^3")

    groups = passage_groups([category, dosimeter, radon], [SECTION_75, SECTION_91])

    assert groups == [["category", "dosimeter"], ["radon"]]


def test_the_same_wording_in_different_chunks_does_not_join_items():
    chunks = [
        "Intervallet mellem statuskontroller må ikke være længere end 25 måneder.",
        "Intervallet mellem eftersyn af kilden må ikke være længere end 25 måneder.",
    ]
    status = _item("status", "statuskontroller må ikke være længere end 25 måneder")
    inspection = _item("inspection", "af kilden må ikke være længere end 25 måneder")

    assert passage_groups([status, inspection], chunks) == [["status"], ["inspection"]]


def _own_chunks(items):
    """One chunk per item, holding exactly its quotes: no item shares a chunk."""
    return [
        " ".join(q for n in item["nuggets"] for q in n["evidence"]) for item in items
    ]


def _splits(items):
    return {item["id"]: item["split"] for item in items}


def test_each_stratum_is_divided_evenly_between_tune_and_confirm():
    items = [_item(f"dk{n}", f"dansk regel nummer {n}") for n in range(4)] + [
        _item(f"iaea{n}", f"iaea rule number {n}", source="iaea", difficulty="easy")
        for n in range(4)
    ]

    assigned = assign_splits(items, _own_chunks(items))

    per_stratum = Counter((i["source"], i["split"]) for i in assigned)
    assert per_stratum == {
        ("dk-law", "tune"): 2,
        ("dk-law", "confirm"): 2,
        ("iaea", "tune"): 2,
        ("iaea", "confirm"): 2,
    }


def test_a_chunk_the_development_set_quotes_puts_its_whole_group_into_tune():
    chunks = [SECTION_75, SECTION_91, "§ 92. Arbejdspladser 100 Bq/m^3."]
    category = _item("category", "i kategori A og kategori B skal være omfattet")
    dosimeter = _item("dosimeter", "måleperiode på højst 3 måneder")
    radon = _item("radon", "radon i bygninger er 100 Bq/m^3")
    workplace = _item("workplace", "Arbejdspladser 100 Bq/m^3")
    developed_on = [_item("dev-dosimetry", "Stråleudsatte arbejdstagere i kategori A")]

    assigned = assign_splits(
        [category, dosimeter, radon, workplace], chunks, dev_items=developed_on
    )

    assert _splits(assigned)["category"] == "tune"
    assert _splits(assigned)["dosimeter"] == "tune"


def test_assigned_items_keep_their_split_and_new_items_join_their_chunk():
    chunks = [SECTION_75, SECTION_91, "§ 92. Arbejdspladser 100 Bq/m^3."]
    old_category = _item("old-category", "i kategori A og kategori B", split="confirm")
    old_radon = _item("old-radon", "radon i bygninger er 100 Bq/m^3", split="tune")
    new_dosimeter = _item("new-dosimeter", "måleperiode på højst 3 måneder")
    new_workplace = _item("new-workplace", "Arbejdspladser 100 Bq/m^3")

    assigned = assign_splits(
        [old_category, old_radon, new_dosimeter, new_workplace], chunks
    )

    splits = _splits(assigned)
    assert (splits["old-category"], splits["old-radon"]) == ("confirm", "tune")
    assert splits["new-dosimeter"] == "confirm"


def test_a_chunk_shared_across_both_splits_is_an_error_not_a_silent_move():
    tune_item = _item("in-tune", "i kategori A og kategori B", split="tune")
    confirm_item = _item(
        "in-confirm", "måleperiode på højst 3 måneder", split="confirm"
    )

    with pytest.raises(ValueError, match="in-tune.*in-confirm"):
        assign_splits([tune_item, confirm_item], [SECTION_75])


def test_selecting_a_split_keeps_only_its_items_in_file_order():
    items = [
        {"id": "a", "split": "tune"},
        {"id": "b"},
        {"id": "c", "split": "confirm"},
        {"id": "d", "split": "tune"},
    ]

    assert [i["id"] for i in select_split(items, "tune")] == ["a", "d"]
    assert select_split(items, "missing") == []


def test_leaks_name_items_that_share_a_chunk_across_splits_or_with_the_dev_set():
    chunks = [SECTION_75, SECTION_91]
    items = [
        _item("in-tune", "i kategori A og kategori B", split="tune"),
        _item("in-confirm", "måleperiode på højst 3 måneder", split="confirm"),
        _item("radon", "radon i bygninger er 100 Bq/m^3", split="confirm"),
    ]
    developed_on = [_item("dev-radon", "Referenceniveauet for radon")]

    assert split_leaks(items, chunks, developed_on) == [
        "in-tune (tune) and in-confirm (confirm) share an index chunk",
        "radon (confirm) shares an index chunk with the development set",
    ]


def test_a_clean_split_has_no_leaks():
    chunks = [SECTION_75, SECTION_91]
    items = [
        _item("category", "i kategori A og kategori B", split="tune"),
        _item("dosimeter", "måleperiode på højst 3 måneder", split="tune"),
        _item("radon", "radon i bygninger er 100 Bq/m^3", split="confirm"),
    ]

    assert split_leaks(items, chunks, [_item("dev", "kategori A og kategori B")]) == []


def _cli(monkeypatch, tmp_path, items, dev, chunks, *extra):
    golden, dev_file = tmp_path / "golden_test.json", tmp_path / "golden.json"
    golden.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    dev_file.write_text(json.dumps(dev, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("graph.llm_factory.get_embedding_provider", lambda: "scaleway")
    monkeypatch.setattr("ingestion.load_chunk_texts", lambda provider: {"dk": chunks})
    argv = ["eval.split", "--golden", str(golden), "--dev", str(dev_file), *extra]
    monkeypatch.setattr("sys.argv", argv)
    return split_cli.main(), json.loads(golden.read_text(encoding="utf-8"))


def test_the_command_writes_a_split_for_every_item_from_the_real_index(
    monkeypatch, tmp_path, capsys
):
    items = [
        _item("category", "i kategori A og kategori B"),
        _item("dosimeter", "måleperiode på højst 3 måneder"),
        _item("radon", "radon i bygninger er 100 Bq/m^3"),
    ]
    dev = [_item("dev-radon", "Referenceniveauet for radon")]

    code, written = _cli(monkeypatch, tmp_path, items, dev, [SECTION_75, SECTION_91])

    assert code == 0
    assert _splits(written)["radon"] == "tune"
    assert _splits(written)["category"] == _splits(written)["dosimeter"]
    assert "tune: " in capsys.readouterr().out


def test_the_index_check_reports_a_split_leak_as_an_error(
    monkeypatch, tmp_path, capsys
):
    items = [
        _item("in-tune", "i kategori A og kategori B", split="tune"),
        _item("in-confirm", "måleperiode på højst 3 måneder", split="confirm"),
    ]
    golden, dev_file = tmp_path / "golden_test.json", tmp_path / "golden.json"
    golden.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    dev_file.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(golden_cli, "index_chunks", lambda: ([SECTION_75], ["dk"]))
    argv = ["eval.golden", "--golden", str(golden), "--dev", str(dev_file)]
    monkeypatch.setattr("sys.argv", [*argv, "--check-index"])

    assert golden_cli.main() == 1
    assert (
        "Error: in-tune (tune) and in-confirm (confirm) share an index chunk"
        in capsys.readouterr().out
    )
