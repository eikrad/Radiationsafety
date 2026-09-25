"""Tests for judge calibration fixtures and the judge_check tool (fake judge)."""

import json
from pathlib import Path

import pytest

from eval import judge_check
from eval.judge_check import DEFAULT_FIXTURES, check_judge, load_fixtures

FIXTURE = {
    "id": "wrong-unit-gy",
    "purpose": "Gy instead of Sv must not count as the fact",
    "origin": "radiation-specific",
    "question": "Hvad er dosisgrænsen?",
    "expected_behavior": "answer",
    "nuggets": [
        {"text": "20 mSv pr. år", "importance": "vital", "evidence": ["20 mSv pr. år"]}
    ],
    "context": "Dosisgrænsen for effektiv dosis er 20 mSv pr. år.",
    "answer": "Dosisgrænsen er 20 mGy pr. år.",
    "expect": {
        "nuggets": [["not_support", "partial_support"]],
        "unsupported": True,
        "refused": False,
    },
}


def _judge_returning(verdict):
    return lambda item, answer, context, llm: verdict


def test_a_judge_that_catches_the_error_passes_the_fixture(monkeypatch):
    monkeypatch.setattr(
        judge_check,
        "judge_item",
        _judge_returning(
            {
                "nuggets": ["not_support"],
                "unsupported_claims": ["20 mGy"],
                "refused": False,
            }
        ),
    )

    [result] = check_judge([FIXTURE], llm=None)

    assert result == {"id": "wrong-unit-gy", "passed": True, "problems": []}


def test_a_judge_that_misses_the_unit_error_fails_with_the_reasons(monkeypatch):
    monkeypatch.setattr(
        judge_check,
        "judge_item",
        _judge_returning(
            {"nuggets": ["support"], "unsupported_claims": [], "refused": False}
        ),
    )

    [result] = check_judge([FIXTURE], llm=None)

    assert result["passed"] is False
    assert result["problems"] == [
        "nugget 1: got support, expected not_support or partial_support",
        "unsupported claims: got none, expected at least one",
    ]


def test_a_judge_that_returns_nothing_fails_the_fixture(monkeypatch):
    monkeypatch.setattr(judge_check, "judge_item", _judge_returning(None))

    [result] = check_judge([FIXTURE], llm=None)

    assert result["problems"] == ["judge returned no verdict"]


def test_the_bundled_fixtures_are_well_formed():
    fixtures = load_fixtures(DEFAULT_FIXTURES)

    assert len(fixtures) >= 15
    assert len({f["id"] for f in fixtures}) == len(fixtures)
    for fixture in fixtures:
        assert len(fixture["expect"]["nuggets"]) == len(fixture["nuggets"]), fixture[
            "id"
        ]
        assert fixture["purpose"] and fixture["context"] and fixture["answer"]


def test_a_fixture_with_mismatched_expectations_is_rejected(tmp_path):
    broken = {**FIXTURE, "expect": {**FIXTURE["expect"], "nuggets": []}}
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps([broken]), encoding="utf-8")

    with pytest.raises(
        ValueError, match="wrong-unit-gy: 1 nuggets but 0 expected labels"
    ):
        load_fixtures(path)


def test_the_command_reports_a_pass_rate_per_judge(monkeypatch, tmp_path, capsys):
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps([FIXTURE]), encoding="utf-8")
    monkeypatch.setattr(
        judge_check,
        "judge_item",
        _judge_returning(
            {"nuggets": ["not_support"], "unsupported_claims": ["x"], "refused": False}
        ),
    )
    monkeypatch.setattr(judge_check, "_make_judge", lambda spec: object())
    monkeypatch.setattr(
        "sys.argv",
        ["judge_check", "--fixtures", str(path), "--judge", "scaleway:model-a"],
    )

    assert judge_check.main() == 0

    out = capsys.readouterr().out
    assert "scaleway:model-a" in out
    assert "1/1 fixtures passed" in out
    assert Path(DEFAULT_FIXTURES).exists()
