"""What a recorded run cost: its tokens at the prices valid on the day it ran.

A run records tokens per stage and model (`usage` in eval/history.py), not
money: prices change. eval/data/prices.json holds prices per million tokens,
one entry per model and the date from which it applies (`since`). When a price
changes, add an entry instead of editing the old one, so past runs keep the
price they were run at.
"""

import json
from pathlib import Path

DEFAULT_PRICES = Path(__file__).resolve().parent / "data" / "prices.json"


def load_prices(path: Path = DEFAULT_PRICES) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cost(run: dict, prices: list[dict]) -> dict | None:
    """Cost of a run per stage and in total, per currency; None without tokens.

    A model with no price on the run's day is listed under ``unpriced`` and
    left out of the sums, never counted as free.
    """
    usage = run.get("usage")
    if not usage:
        return None
    day = _day(run["run_id"])
    by_stage: dict[str, dict[str, float]] = {}
    total: dict[str, float] = {}
    unpriced: list[str] = []
    for stage, models in usage.items():
        stage_cost: dict[str, float] = {}
        for model, used in models.items():
            price = _price(prices, model, day)
            if price is None:
                if model not in unpriced:
                    unpriced.append(model)
                continue
            amount, currency = _amount(used, price), price["currency"]
            stage_cost[currency] = stage_cost.get(currency, 0.0) + amount
            total[currency] = total.get(currency, 0.0) + amount
        by_stage[stage] = stage_cost
    return {"by_stage": by_stage, "total": total, "unpriced": unpriced}


def _day(run_id: str) -> str:
    """Run ids start with the UTC date, e.g. 20261014_090000 -> 2026-10-14."""
    return f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"


def _price(prices: list[dict], model: str, day: str) -> dict | None:
    valid = [p for p in prices if p["model"] == model and p["since"] <= day]
    return max(valid, key=lambda p: p["since"]) if valid else None


def _amount(used: dict, price: dict) -> float:
    cached = (used.get("input_token_details") or {}).get("cache_read") or 0
    fresh = used.get("input_tokens", 0) - cached
    per_million = (
        fresh * price["input"]
        + cached * price.get("cached_input", price["input"])
        + used.get("output_tokens", 0) * price["output"]
    )
    return per_million / 1_000_000
