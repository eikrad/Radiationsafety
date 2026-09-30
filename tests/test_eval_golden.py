"""Tests for the golden set format v2: nuggets with evidence, refusal items."""

import json

import pytest

from eval.golden import (
    GoldenError,
    check_evidence_in_index,
    golden_warnings,
    load_golden,
)

ANSWERABLE = {
    "id": "dk-dose-limits",
    "question": "Hvor findes dosisgrænserne for erhvervsmæssig bestråling?",
    "topics": ["occupational"],
    "language": "da",
    "source": "dk-law",
    "expected_behavior": "answer",
    "nuggets": [
        {
            "text": "Dose limits for occupational exposure are set out in Annex 2",
            "importance": "vital",
            "evidence": [
                "dosisgrænser for erhvervsmæssig bestråling fremgår af bilag 2"
            ],
        },
        {"text": "The limits apply per calendar year", "importance": "okay"},
    ],
}

REFUSAL = {
    "id": "out-of-scope-mri",
    "question": "What is the SAR limit for a 3 T MRI scanner?",
    "topics": ["medical"],
    "language": "en",
    "source": "iaea",
    "expected_behavior": "refuse",
}


def _write(tmp_path, items) -> str:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def _problems(tmp_path, items) -> str:
    with pytest.raises(GoldenError) as err:
        load_golden(_write(tmp_path, items))
    return str(err.value)


def test_an_answerable_item_with_evidenced_nuggets_loads(tmp_path):
    [item] = load_golden(_write(tmp_path, [ANSWERABLE]))

    assert item["expected_behavior"] == "answer"
    vital, okay = item["nuggets"]
    assert vital["importance"] == "vital"
    assert okay["evidence"] == []


def test_a_refusal_item_needs_no_nuggets(tmp_path):
    [item] = load_golden(_write(tmp_path, [REFUSAL]))

    assert item["expected_behavior"] == "refuse"
    assert item["nuggets"] == []


def test_expected_behavior_defaults_to_answer(tmp_path):
    item = {k: v for k, v in ANSWERABLE.items() if k != "expected_behavior"}

    [loaded] = load_golden(_write(tmp_path, [item]))

    assert loaded["expected_behavior"] == "answer"


def test_an_answerable_item_without_a_vital_nugget_is_rejected(tmp_path):
    item = {**ANSWERABLE, "nuggets": [ANSWERABLE["nuggets"][1]]}

    assert "dk-dose-limits: needs at least one vital nugget" in _problems(
        tmp_path, [item]
    )


def test_a_vital_nugget_without_evidence_is_rejected(tmp_path):
    vital = {**ANSWERABLE["nuggets"][0], "evidence": []}
    item = {**ANSWERABLE, "nuggets": [vital]}

    assert "dk-dose-limits: vital nugget 1 has no evidence quote" in _problems(
        tmp_path, [item]
    )


def test_a_refusal_item_with_nuggets_is_rejected(tmp_path):
    item = {**REFUSAL, "nuggets": ANSWERABLE["nuggets"]}

    assert "out-of-scope-mri: refuse items must not have nuggets" in _problems(
        tmp_path, [item]
    )


def test_a_misspelled_topic_is_rejected_instead_of_creating_a_new_group(tmp_path):
    item = {**ANSWERABLE, "topics": ["ocupational"]}

    assert "dk-dose-limits: unknown topic 'ocupational'" in _problems(tmp_path, [item])


def test_every_problem_is_reported_at_once(tmp_path):
    broken_nugget = {"text": "", "importance": "crucial", "evidence": [""]}
    items = [
        {**ANSWERABLE, "nuggets": [ANSWERABLE["nuggets"][0], broken_nugget]},
        {**REFUSAL, "id": "dk-dose-limits", "expected_behavior": "maybe"},
    ]

    problems = _problems(tmp_path, items)

    assert "dk-dose-limits: nugget 2 has no text" in problems
    assert "dk-dose-limits: nugget 2 has importance 'crucial'" in problems
    assert "dk-dose-limits: nugget 2 has an empty evidence quote" in problems
    assert "dk-dose-limits: duplicate id" in problems
    assert "dk-dose-limits: expected_behavior 'maybe'" in problems


def test_legacy_key_facts_are_rejected_with_a_migration_hint(tmp_path):
    item = {"id": "old", "question": "Hvad?", "key_facts": ["bilag 2"]}

    assert "old: key_facts is the v1 format; use nuggets" in _problems(tmp_path, [item])


def test_the_file_must_be_a_list_of_items(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text('{"id": "x"}', encoding="utf-8")

    with pytest.raises(GoldenError, match="JSON array"):
        load_golden(path)


def test_long_evidence_quotes_load_but_are_flagged(tmp_path):
    long_quote = "x " * 100
    vital = {**ANSWERABLE["nuggets"][0], "evidence": [long_quote]}
    items = load_golden(_write(tmp_path, [{**ANSWERABLE, "nuggets": [vital]}]))

    [warning] = golden_warnings(items)

    assert warning.startswith("dk-dose-limits: vital nugget 1 evidence quote is 200")
    assert "chunk boundary" in warning


# --- evidence quotes against the search index ---------------------------------

TWO_QUOTES = {
    **ANSWERABLE,
    "nuggets": [
        {
            "text": "Annex 2",
            "importance": "vital",
            "evidence": ["fremgår af bilag 2", "anført i bilag 2"],
        },
        {"text": "per year", "importance": "okay", "evidence": ["pr. kalenderår"]},
    ],
}


def test_a_vital_nugget_whose_quotes_are_in_no_chunk_is_an_error():
    errors, _ = check_evidence_in_index([TWO_QUOTES], ["nothing relevant here"])

    assert errors == [
        "dk-dose-limits: vital nugget 1: no evidence quote is in any chunk "
        "(misquoted, or split across a chunk boundary)"
    ]


def test_one_matching_alternative_is_enough_but_the_others_are_reported():
    chunks = ["Grænserne fremgår af Bilag 2 og gælder pr. kalenderår."]

    errors, warnings = check_evidence_in_index([TWO_QUOTES], chunks)

    assert errors == []
    assert warnings == [
        "dk-dose-limits: vital nugget 1: quote 'anført i bilag 2' is in no chunk"
    ]


def test_a_quote_found_in_many_chunks_is_reported_as_unspecific():
    chunks = ["fremgår af bilag 2 (a)"] * 4 + ["anført i bilag 2", "pr. kalenderår"]

    _, warnings = check_evidence_in_index([TWO_QUOTES], chunks)

    assert warnings == [
        "dk-dose-limits: vital nugget 1: quote 'fremgår af bilag 2' is in 4 chunks; "
        "a more specific quote tells the right passage apart"
    ]


def test_a_missing_quote_for_an_okay_nugget_is_only_a_warning():
    chunks = ["fremgår af bilag 2", "anført i bilag 2"]

    errors, warnings = check_evidence_in_index([TWO_QUOTES], chunks)

    assert errors == []
    assert warnings == [
        "dk-dose-limits: okay nugget 2: quote 'pr. kalenderår' is in no chunk"
    ]


def test_questions_to_refuse_have_nothing_to_check():
    assert check_evidence_in_index([REFUSAL], []) == ([], [])
