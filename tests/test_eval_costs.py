"""What a recorded run cost, from its tokens and the prices valid on its day."""

import re

import pytest

from eval.costs import load_prices, run_cost

PRICES = [
    {
        "model": "gemma",
        "currency": "EUR",
        "since": "2026-10-01",
        "input": 0.25,
        "output": 0.50,
    },
    {
        "model": "judge",
        "currency": "EUR",
        "since": "2026-10-01",
        "input": 0.60,
        "cached_input": 0.12,
        "output": 3.30,
    },
    {
        "model": "flash",
        "currency": "USD",
        "since": "2026-10-01",
        "input": 0.75,
        "output": 3.75,
    },
    {
        "model": "flash",
        "currency": "USD",
        "since": "2027-01-01",
        "input": 1.50,
        "output": 7.50,
    },
]


def _run(run_id, **usage):
    return {"run_id": run_id, "usage": usage}


def _tokens(tokens_in, tokens_out, cached=0):
    used = {"input_tokens": tokens_in, "output_tokens": tokens_out}
    if cached:
        used["input_token_details"] = {"cache_read": cached}
    return used


def test_a_run_costs_its_tokens_at_the_prices_per_million():
    run = _run(
        "20261014_090000",
        answers={"gemma": _tokens(2_000_000, 100_000)},
        judge={"judge": _tokens(1_000_000, 200_000)},
    )

    cost = run_cost(run, PRICES)

    assert cost["by_stage"]["answers"] == {"EUR": pytest.approx(0.55)}
    assert cost["by_stage"]["judge"] == {"EUR": pytest.approx(1.26)}
    assert cost["total"] == {"EUR": pytest.approx(1.81)}
    assert cost["unpriced"] == []


def test_cached_input_is_charged_at_the_cached_price():
    run = _run(
        "20261014_090000", judge={"judge": _tokens(1_000_000, 0, cached=500_000)}
    )

    assert run_cost(run, PRICES)["total"] == {"EUR": pytest.approx(0.36)}


def test_the_price_valid_on_the_day_of_the_run_applies():
    december = _run("20261215_090000", answers={"flash": _tokens(1_000_000, 0)})
    january = _run("20270105_090000", answers={"flash": _tokens(1_000_000, 0)})

    assert run_cost(december, PRICES)["total"] == {"USD": pytest.approx(0.75)}
    assert run_cost(january, PRICES)["total"] == {"USD": pytest.approx(1.50)}


def test_a_model_without_a_price_is_named_not_counted_as_free():
    run = _run("20261014_090000", answers={"unknown-model": _tokens(10, 10)})

    cost = run_cost(run, PRICES)

    assert cost["unpriced"] == ["unknown-model"]
    assert cost["total"] == {}


def test_a_run_without_recorded_tokens_has_no_cost():
    assert run_cost({"run_id": "20261011_061712"}, PRICES) is None


def test_the_price_table_names_model_currency_day_and_rates():
    prices = load_prices()

    for price in prices:
        assert price["currency"] in {"EUR", "USD"}, price
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", price["since"]), price
        assert price["input"] >= 0 and price["output"] >= 0, price
        assert price.get("source"), price
    entries = [(p["model"], p["since"]) for p in prices]
    assert len(entries) == len(set(entries)), "one price per model and day"
