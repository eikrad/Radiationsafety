"""The held-out test set: format, tune/confirm split and separation from the dev set.

That no chunk of the search index is shared across the splits, or between
confirm and the development set, needs the index:
`uv run python -m eval.golden --golden eval/data/golden_test.json --check-index`.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from eval.golden import load_golden
from eval.split import SPLITS, select_split

_DATA = Path(__file__).resolve().parents[1] / "eval" / "data"


@pytest.fixture(scope="module")
def test_set() -> list[dict]:
    return load_golden(_DATA / "golden_test.json")


def test_every_item_is_in_tune_or_confirm(test_set):
    assert Counter(item.get("split") for item in test_set).keys() == set(SPLITS)


def test_the_splits_are_about_the_same_size(test_set):
    tune, confirm = (len(select_split(test_set, s)) for s in SPLITS)
    assert abs(tune - confirm) <= 4


def test_both_splits_hold_items_to_refuse(test_set):
    """The scope check (#150) is tuned on tune and confirmed on confirm."""
    refusals = Counter(
        i["split"] for i in test_set if i["expected_behavior"] == "refuse"
    )
    assert all(refusals[s] >= 2 for s in SPLITS)


@pytest.mark.parametrize("tag", ["source", "language", "difficulty"])
def test_both_splits_cover_every_value_of_a_tag(test_set, tag):
    for split in SPLITS:
        values = {i[tag] for i in select_split(test_set, split)}
        assert values == {i[tag] for i in test_set}, split


def test_both_splits_cover_every_topic(test_set):
    for split in SPLITS:
        topics = {t for i in select_split(test_set, split) for t in i["topics"]}
        assert topics == {t for i in test_set for t in i["topics"]}, split


def test_evidence_quotes_use_plain_quotation_marks(test_set):
    """The index has ' and ", not typographic quotes: such a quote never matches."""
    curly = [
        item["id"]
        for item in test_set
        for nugget in item["nuggets"]
        for quote in nugget["evidence"]
        if re.search("[‘’“”]", quote)
    ]
    assert not curly


def test_test_set_shares_no_id_with_the_development_set(test_set):
    development = {i["id"] for i in load_golden(_DATA / "golden.json")}
    assert development.isdisjoint(i["id"] for i in test_set)


# A user does not know where an answer is written, so a question must not point to
# a paragraph, annex, table, section or a named standard or order. The legal
# system (Danish rules, IAEA safety standards) may be named: the golden-set rule
# for questions where the two give different answers.
_LOCATOR = re.compile(
    r"(§|\bstk\.|\bbilag\b|\bannex\b|\bsection\s*\d|\btable\s*\d|\btabel\b"
    r"|\bappendix\b|\bparas?\.?\s*\d|\bGSR\b|\bGSG\b|\bSSG\b|\bSSR\b|TECDOC"
    r"|\b24-?G\b|bekendtgørelse|\border on\b|\bDanish order\b|\bthe order\b|\bBEK\b)",
    re.IGNORECASE,
)


def test_questions_do_not_say_where_to_look(test_set):
    pointing = [item["id"] for item in test_set if _LOCATOR.search(item["question"])]
    assert not pointing
