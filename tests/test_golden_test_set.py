"""The held-out test set: format, tiers and separation from the development set."""

from pathlib import Path

import pytest

from eval.golden import load_golden, select_tier

_DATA = Path(__file__).resolve().parents[1] / "eval" / "data"
TIERS = {"core", "extended"}


@pytest.fixture(scope="module")
def test_set() -> list[dict]:
    return load_golden(_DATA / "golden_test.json")


def test_every_item_has_a_known_tier(test_set):
    assert {item.get("tier") for item in test_set} <= TIERS


def test_tiers_partition_the_set(test_set):
    core = select_tier(test_set, "core")
    extended = select_tier(test_set, "extended")
    assert len(core) + len(extended) == len(test_set)
    assert {i["id"] for i in core}.isdisjoint(i["id"] for i in extended)


def test_core_keeps_the_items_to_refuse(test_set):
    """Refusal checks are cheap and the scope check (#150) needs them every run."""
    refusals = [i for i in test_set if i["expected_behavior"] == "refuse"]
    assert refusals
    assert all(i["tier"] == "core" for i in refusals)


def test_core_has_no_easy_answerable_item(test_set):
    easy = [
        i["id"]
        for i in select_tier(test_set, "core")
        if i["expected_behavior"] == "answer" and i.get("difficulty") == "easy"
    ]
    assert not easy


def test_core_covers_every_source(test_set):
    assert {i["source"] for i in select_tier(test_set, "core")} == {
        "dk-law",
        "iaea",
        "both",
    }


def test_test_set_shares_no_id_with_the_development_set(test_set):
    development = {i["id"] for i in load_golden(_DATA / "golden.json")}
    assert development.isdisjoint(i["id"] for i in test_set)


def test_select_tier_ignores_untagged_items():
    items = [{"id": "a", "tier": "core"}, {"id": "b"}, {"id": "c", "tier": "extended"}]
    assert [i["id"] for i in select_tier(items, "core")] == ["a"]
    assert select_tier(items, "missing") == []
