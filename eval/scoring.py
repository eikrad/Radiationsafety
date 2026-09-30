"""Scoring v2: deterministic evidence metrics and per-question error attribution.

The LLM judge only answers two narrow questions (which nuggets does the answer
contain; which claims are unsupported / did it refuse). Everything else is
derived here without tokens:

- evidence_recall_initial / _context: share of vital nuggets whose evidence
  quote appears in the first retrieval / in the context the generator saw
  (BEIR-style retrieval metric; quotes instead of chunk labels, so re-chunking
  does not invalidate the golden set).
- vital_recall (strict), vital_recall_lenient (partial = 0.5), all_recall:
  nugget recall of the answer (Pradeep et al. 2025, V_strict / A_strict).
- grounded_vital_recall: vital nuggets that are both in the answer and
  evidenced in the context. Facts from model pretraining do not count
  (Trust-Score, Song et al. 2025).
- context_utilization: of the vital nuggets evidenced in the context, how many
  the answer used (RAGChecker, Ru et al. 2024). This is a generator-side diagnostic.
- error_type: why a question failed, attributed to retrieval or generation.
- evidence_position: which chunk of the generator context first holds each
  vital nugget's evidence (Lost in the Middle, Liu et al. 2024: position can
  matter as much as presence).
- warning_shown: whether the answer carried a warning for the user.

Rank-based retrieval metrics (retrieval-only runs, which retrieve deeper than
the graph): evidence_recall_at_<k>, evidence_recall_budget (equal amount of
text, so chunkings with different chunk lengths compare fairly; Dense X, Chen
et al. 2024) and reciprocal_rank (its mean is MRR).
"""

import re
import unicodedata
from collections.abc import Iterable

from langchain_core.documents import Document

from graph.consts import CONTEXT_SEPARATOR

# Bump when a metric's definition changes; the dashboard then marks runs on
# either side as not directly comparable. 1 = RAGAS-style binary metrics
# (eval/metrics.py, removed; see git history), 2 = evidence + nugget scoring
# defined in this module.
METRICS_VERSION = 2

# Depths at which retrieval-only runs report evidence recall.
RECALL_DEPTHS = (1, 3, 5, 10, 20)

SUPPORT = "support"
PARTIAL = "partial_support"
NOT_SUPPORT = "not_support"

# Hyphen and dash variants (U+2010-U+2015, minus sign U+2212) all match "-".
# Code points, not literal characters, so the table survives formatters.
_DASHES = dict.fromkeys([*range(0x2010, 0x2016), 0x2212], "-")


def normalize(text: str) -> str:
    """Case-, whitespace- and typography-insensitive form for verbatim matching.

    Invisible format characters (Unicode category Cf) are dropped: the Danish
    XML sources contain soft hyphens and zero-width joiners inside words.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.translate(_DASHES).casefold()
    return re.sub(r"\s+", " ", text).strip()


def evidence_found(quotes: Iterable[str], texts: Iterable[str]) -> bool:
    """True if any quote appears verbatim (after normalisation) in any text."""
    haystacks = [normalize(t) for t in texts]
    return any(normalize(q) in h for q in quotes for h in haystacks)


def _texts(documents: Iterable[Document | str]) -> list[str]:
    return [d if isinstance(d, str) else d.page_content for d in documents]


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def score_item(item: dict, output: dict, verdict: dict | None) -> dict:
    """Score one golden item.

    output: {"initial_documents": [...], "context": str, "sufficient": bool | None}
        from the graph run (first retrieval, generator context, grade_documents).
    verdict: {"nuggets": [label per nugget], "unsupported_claims": [...],
        "refused": bool} from the judge, or None when the judge failed.
    """
    nuggets = item.get("nuggets") or []
    if verdict is not None and len(verdict["nuggets"]) != len(nuggets):
        raise ValueError(
            f"{item.get('id')}: {len(nuggets)} nuggets but "
            f"{len(verdict['nuggets'])} labels in the verdict"
        )

    initial_texts = _texts(output.get("initial_documents") or [])
    context_texts = [output.get("context") or ""]
    vital = [i for i, n in enumerate(nuggets) if n["importance"] == "vital"]
    in_initial = {
        i: evidence_found(nuggets[i]["evidence"], initial_texts) for i in vital
    }
    in_context = {
        i: evidence_found(nuggets[i]["evidence"], context_texts) for i in vital
    }
    answerable = item["expected_behavior"] == "answer"

    scores = {
        "evidence_recall_initial": _share(list(in_initial.values())),
        "evidence_recall_context": _share(list(in_context.values())),
        "grade_documents_correct": _grader_correct(
            output.get("sufficient"), answerable, in_initial
        ),
        "vital_recall": None,
        "vital_recall_lenient": None,
        "all_recall": None,
        "grounded_vital_recall": None,
        "context_utilization": None,
        "unsupported_claims": None,
        "refused": None,
        "evidence_position": _evidence_position(
            [nuggets[i] for i in vital], output.get("context") or ""
        ),
        "warning_shown": bool(output.get("retrieval_warning")),
    }
    if verdict is None:
        return {**scores, "error_type": "judge_error", "pass": None}

    labels = verdict["nuggets"]
    unsupported = len(verdict.get("unsupported_claims") or [])
    refused = bool(verdict.get("refused"))
    scores["unsupported_claims"] = unsupported
    scores["refused"] = refused

    if answerable:
        supported = {i: labels[i] == SUPPORT for i in range(len(nuggets))}
        evidenced = [i for i in vital if in_context[i]]
        scores.update(
            vital_recall=_share([supported[i] for i in vital]),
            vital_recall_lenient=sum(_credit(labels[i]) for i in vital) / len(vital),
            all_recall=_share(list(supported.values())),
            grounded_vital_recall=_share(
                [supported[i] and in_context[i] for i in vital]
            ),
            context_utilization=_share([supported[i] for i in evidenced]),
        )
        error_type = _answer_error(
            all_vital_supported=all(supported[i] for i in vital),
            all_vital_evidenced=len(evidenced) == len(vital),
            unsupported=unsupported,
            refused=refused,
        )
    else:
        error_type = "ok" if refused and not unsupported else "missed_refusal"

    return {**scores, "error_type": error_type, "pass": error_type == "ok"}


def _credit(label: str) -> float:
    return {SUPPORT: 1.0, PARTIAL: 0.5}.get(label, 0.0)


def _answer_error(
    *,
    all_vital_supported: bool,
    all_vital_evidenced: bool,
    unsupported: int,
    refused: bool,
) -> str:
    """Why an answerable question failed, blaming retrieval before generation.

    If a vital fact's evidence never reached the generator, nothing the
    generator does is grounded, and refusing is even the right reaction
    (answerability is relative to the retrieved documents, as in Trust-Score).
    """
    if all_vital_supported and all_vital_evidenced and not unsupported:
        return "ok"
    if not all_vital_evidenced:
        return "retrieval_miss"
    if unsupported:
        return "unsupported_claim"
    if refused:
        return "wrong_refusal"
    return "generator_miss"


def _grader_correct(
    sufficient: bool | None, answerable: bool, in_initial: dict[int, bool]
) -> bool | None:
    """Did grade_documents judge the first retrieval correctly?

    Sufficient is right exactly when every vital nugget's evidence was retrieved;
    for questions that should be refused, "insufficient" is right.
    """
    if sufficient is None:
        return None
    truly_sufficient = answerable and all(in_initial.values())
    return sufficient == truly_sufficient


def _evidence_position(vital_nuggets: list[dict], context: str) -> float | None:
    """Mean 1-based position of the first context chunk holding each vital
    nugget's evidence, over the nuggets whose evidence is in the context."""
    chunks = context.split(CONTEXT_SEPARATOR) if context else []
    positions = [
        next(
            (p for p, chunk in enumerate(chunks, 1) if evidence_found(quotes, [chunk])),
            None,
        )
        for quotes in (n["evidence"] for n in vital_nuggets)
    ]
    found = [p for p in positions if p is not None]
    return sum(found) / len(found) if found else None


def evidence_ranks(
    item: dict, ranked: dict[str, list[Document | str]]
) -> list[int | None]:
    """For each vital nugget, the 1-based rank of the first chunk holding its
    evidence, within the collection where it ranks best; None if not retrieved.

    ranked: each collection's retrieval in rank order. Ranks are per collection
    because the generator receives every collection's list in full.
    """
    ranks = []
    for nugget in item.get("nuggets") or []:
        if nugget["importance"] != "vital":
            continue
        found = [
            next(
                (
                    r
                    for r, text in enumerate(_texts(docs), 1)
                    if evidence_found(nugget["evidence"], [text])
                ),
                None,
            )
            for docs in ranked.values()
        ]
        hits = [r for r in found if r is not None]
        ranks.append(min(hits) if hits else None)
    return ranks


def ranking_metrics(
    item: dict,
    ranked: dict[str, list[Document | str]],
    *,
    depth: int,
    char_budget: int | None,
) -> dict[str, float]:
    """Rank-based evidence metrics for one question; {} if it has no vital nugget.

    depth: how many chunks per collection were retrieved; recall is reported
    for the RECALL_DEPTHS up to it. char_budget: characters per collection;
    chunks count in rank order while the running total fits.
    """
    ranks = evidence_ranks(item, ranked)
    if not ranks:
        return {}
    metrics = {
        f"evidence_recall_at_{k}": _share([r is not None and r <= k for r in ranks])
        for k in RECALL_DEPTHS
        if k <= depth
    }
    metrics["reciprocal_rank"] = sum(1 / r if r else 0.0 for r in ranks) / len(ranks)
    if char_budget is not None:
        within = {
            name: _within_budget(docs, char_budget) for name, docs in ranked.items()
        }
        metrics["evidence_recall_budget"] = _share(
            [r is not None for r in evidence_ranks(item, within)]
        )
    return metrics


def _within_budget(docs: list[Document | str], char_budget: int) -> list[str]:
    """The rank-order prefix of docs whose total length fits in char_budget."""
    kept, total = [], 0
    for text in _texts(docs):
        total += len(text)
        if total > char_budget:
            break
        kept.append(text)
    return kept
