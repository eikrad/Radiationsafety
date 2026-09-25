"""Check a judge model against calibration fixtures before trusting its scores.

    uv run python -m eval.judge_check --judge scaleway:<model-id> [--judge ...]

Each fixture (eval/judge_fixtures.json) is a question, nuggets, a context, an
answer and the verdict a good judge must reach. They cover the GroUSE
failure modes (Muller et al. 2024: correct answers, partial answers,
fabrications, correct and wrong refusals, answering from own knowledge, an
absurd context the judge must follow rather than its own knowledge) plus
radiation-specific errors: Gy vs Sv, µSv vs mSv, wrong annex, a foreign rule
instead of the Danish one.

Correlation with a strong model is not enough; a judge must pass these unit
tests (GroUSE). Run this when the judge prompt or model changes; it costs
about two LLM calls per fixture and is not part of CI.
"""

import argparse
import json
import sys
from pathlib import Path

from eval.judge import judge_item

DEFAULT_FIXTURES = Path(__file__).with_name("judge_fixtures.json")


def load_fixtures(path: Path) -> list[dict]:
    """Load fixtures and check that each has one expected label set per nugget."""
    fixtures = json.loads(Path(path).read_text(encoding="utf-8"))
    for fixture in fixtures:
        expected = fixture["expect"]["nuggets"]
        if len(expected) != len(fixture["nuggets"]):
            raise ValueError(
                f"{fixture['id']}: {len(fixture['nuggets'])} nuggets but "
                f"{len(expected)} expected labels"
            )
    return fixtures


def check_judge(fixtures: list[dict], llm) -> list[dict]:
    """Run the judge on every fixture; list what it got wrong."""
    return [_check(fixture, llm) for fixture in fixtures]


def _check(fixture: dict, llm) -> dict:
    verdict = judge_item(fixture, fixture["answer"], fixture["context"], llm)
    if verdict is None:
        return {
            "id": fixture["id"],
            "passed": False,
            "problems": ["judge returned no verdict"],
        }
    expect = fixture["expect"]
    problems = []
    for n, (label, allowed) in enumerate(
        zip(verdict["nuggets"], expect["nuggets"], strict=True), start=1
    ):
        if label not in allowed:
            problems.append(f"nugget {n}: got {label}, expected {' or '.join(allowed)}")
    has_unsupported = bool(verdict["unsupported_claims"])
    if has_unsupported != expect["unsupported"]:
        got = "; ".join(verdict["unsupported_claims"]) or "none"
        wanted = "at least one" if expect["unsupported"] else "none"
        problems.append(f"unsupported claims: got {got}, expected {wanted}")
    if verdict["refused"] != expect["refused"]:
        problems.append(
            f"refused: got {verdict['refused']}, expected {expect['refused']}"
        )
    return {"id": fixture["id"], "passed": not problems, "problems": problems}


def _make_judge(spec: str):
    """provider:model, e.g. scaleway:qwen/qwen3.5-397b-a17b:int4 or gemini:gemini-2.5-pro."""
    from graph.llm_factory import get_llm, scaleway_chat

    provider, _, model = spec.partition(":")
    if provider == "scaleway":
        return scaleway_chat(model)
    return get_llm(provider=provider, model_variant=model or None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check judge models against calibration fixtures."
    )
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        metavar="PROVIDER:MODEL",
        help="Judge to check; repeat to compare several (e.g. scaleway:<model-id>)",
    )
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures)
    for spec in args.judge:
        results = check_judge(fixtures, _make_judge(spec))
        passed = sum(r["passed"] for r in results)
        print(f"\n== {spec}: {passed}/{len(results)} fixtures passed")
        for r in results:
            if not r["passed"]:
                print(f"  ✗ {r['id']}")
                for problem in r["problems"]:
                    print(f"      {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
