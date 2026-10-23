"""Split the held-out set into tune and confirm, one index chunk at a time.

Items whose evidence sits in the same chunk of the search index (a
Danish/English pair, a "both" question and its single-source counterparts,
two questions on one paragraph) always land in the same split: tuning on one
of them tunes the retrieval of the chunk the other is meant to test. Chunks,
not wording, decide: sister regulations repeat phrases such as "må ikke være
længere end 25 måneder" in different paragraphs.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from eval.golden import GoldenError, index_chunks, load_golden
from eval.scoring import normalize

SPLITS = ("tune", "confirm")
# Fixes the order in which groups are placed; changing it reshuffles a fresh split.
SEED = "held-out-2026-10"
_DATA = Path(__file__).resolve().parent / "data"


def select_split(items: list[dict], split: str) -> list[dict]:
    """Items whose ``split`` tag equals ``split``, in file order.

    The tag is not part of the dataset fingerprint, but a run on one split
    covers fewer items and so has its own ``questions_hash``: runs on different
    splits are never compared.
    """
    return [item for item in items if item.get("split") == split]


def assign_splits(
    items: list[dict], chunks: list[str], dev_items: Sequence[dict] = ()
) -> list[dict]:
    """Copies of ``items``, each with a ``split`` (tune or confirm).

    An item that already has a split keeps it, so questions added later never
    move earlier ones; a new item that shares a chunk with an assigned one joins
    its split. Items that share a chunk are placed as one group. A new group
    with a chunk the development set quotes goes to tune: that chunk has shaped
    earlier choices, so confirm stays free of it. The other groups are placed
    largest first, then in an order fixed by SEED, each on the side that keeps
    every stratum (expected behaviour x source x difficulty), language and
    topic most evenly divided between the two splits.
    """
    assigned = [dict(item) for item in items]
    by_id = {item["id"]: item for item in assigned}
    found = evidence_chunks(assigned, chunks)
    developed = set().union(*evidence_chunks(list(dev_items), chunks).values())
    tally: Counter = Counter()

    def place(group: list[dict], split: str) -> None:
        for item in group:
            item["split"] = split
            tally.update((split, key) for key in _keys(item))

    open_groups = []
    for ids in _groups(assigned, found):
        group = [by_id[i] for i in ids]
        fixed = {item["split"] for item in group if item.get("split")}
        if len(fixed) > 1:
            raise ValueError(
                "items share a chunk but sit in different splits (never move an "
                "item after a run; record the leak instead): "
                + ", ".join(
                    f"{i['id']} ({i['split']})" for i in group if i.get("split")
                )
            )
        new = [item for item in group if not item.get("split")]
        tally.update(
            (item["split"], key)
            for item in group
            if item.get("split")
            for key in _keys(item)
        )
        if fixed:
            place(new, fixed.pop())
        elif new:
            open_groups.append(new)
    on_dev_chunks = [
        g for g in open_groups if any(found[i["id"]] & developed for i in g)
    ]
    for group in on_dev_chunks:
        place(group, "tune")
    free = [g for g in open_groups if g not in on_dev_chunks]
    for group in sorted(free, key=lambda g: (-len(g), _order(g))):
        place(
            group,
            min(
                SPLITS, key=lambda s: (_imbalance(tally, group, s), _tiebreak(group, s))
            ),
        )
    return assigned


def split_leaks(
    items: list[dict], chunks: list[str], dev_items: Sequence[dict] = ()
) -> list[str]:
    """Where the split no longer holds against these chunks.

    Re-chunking or new documents can put two questions from different splits
    into one chunk, or a confirm question into a chunk the development set
    quotes. Items are never moved after a run, so a leak is reported, not fixed.
    """
    found = evidence_chunks(items, chunks)
    developed = set().union(*evidence_chunks(list(dev_items), chunks).values())
    tagged = [item for item in items if item.get("split")]
    leaks = [
        f"{a['id']} ({a['split']}) and {b['id']} ({b['split']}) share an index chunk"
        for n, a in enumerate(tagged)
        for b in tagged[n + 1 :]
        if a["split"] != b["split"] and found[a["id"]] & found[b["id"]]
    ]
    leaks += [
        f"{item['id']} (confirm) shares an index chunk with the development set"
        for item in tagged
        if item["split"] == "confirm" and found[item["id"]] & developed
    ]
    return leaks


def passage_groups(items: list[dict], chunks: list[str]) -> list[list[str]]:
    """Ids of the items whose evidence shares a chunk, one list per group.

    Groups are transitive (a shares a chunk with b, b with c: one group) and
    keep the order of ``items``.
    """
    return _groups(items, evidence_chunks(items, chunks))


def evidence_chunks(items: list[dict], chunks: list[str]) -> dict[str, set[int]]:
    """Per item id, the positions of the chunks that contain one of its quotes."""
    haystacks = [normalize(chunk) for chunk in chunks]
    return {
        item["id"]: {
            n
            for quote in _quotes(item)
            for n, haystack in enumerate(haystacks)
            if quote in haystack
        }
        for item in items
    }


def _keys(item: dict) -> list[tuple]:
    stratum = (
        item.get("expected_behavior", "answer"),
        item.get("source"),
        item.get("difficulty"),
    )
    topics = [("topic", topic) for topic in item.get("topics") or []]
    return [("all",), ("stratum", stratum), ("language", item.get("language"))] + topics


def _imbalance(tally: Counter, group: list[dict], split: str) -> int:
    after = tally.copy()
    after.update((split, key) for item in group for key in _keys(item))
    keys = {key for _, key in after}
    return sum(abs(after["tune", key] - after["confirm", key]) for key in keys)


def _order(group: list[dict]) -> str:
    return hashlib.sha256((SEED + min(i["id"] for i in group)).encode()).hexdigest()


def _tiebreak(group: list[dict], split: str) -> bool:
    """On a tie, a fixed coin flip per group decides, so ties do not all go to tune."""
    return (int(_order(group), 16) % 2 == 0) != (split == "tune")


def _groups(items: list[dict], found: dict[str, set[int]]) -> list[list[str]]:
    parent = list(range(len(items)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_item_in: dict[int, int] = {}
    for i, item in enumerate(items):
        for chunk in found[item["id"]]:
            if chunk in first_item_in:
                parent[root(i)] = root(first_item_in[chunk])
            else:
                first_item_in[chunk] = i

    groups: dict[int, list[str]] = {}
    for i, item in enumerate(items):
        groups.setdefault(root(i), []).append(item["id"])
    return list(groups.values())


def _quotes(item: dict) -> list[str]:
    return [
        normalize(quote)
        for nugget in item.get("nuggets") or []
        for quote in nugget.get("evidence") or []
    ]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Give every held-out item without a split a tune or confirm split."
    )
    parser.add_argument("--golden", type=Path, default=_DATA / "golden_test.json")
    parser.add_argument(
        "--dev",
        type=Path,
        default=_DATA / "golden.json",
        help="Development set: items sharing a chunk with it go to tune",
    )
    args = parser.parse_args()
    try:
        items, dev_items = load_golden(args.golden), load_golden(args.dev)
    except GoldenError as err:
        print(err, file=sys.stderr)
        return 1
    index = index_chunks()
    if index is None:
        return 1
    chunks, _ = index
    try:
        assigned = assign_splits(items, chunks, dev_items)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    # write back only the split, so the validator's defaults stay out of the file
    splits = {item["id"]: item["split"] for item in assigned}
    raw = json.loads(args.golden.read_text(encoding="utf-8"))
    for item in raw:
        item["split"] = splits[item["id"]]
    args.golden.write_text(
        json.dumps(raw, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = Counter(splits.values())
    print(", ".join(f"{split}: {counts[split]} items" for split in SPLITS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
