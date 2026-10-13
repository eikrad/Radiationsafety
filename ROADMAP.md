# Evaluation Roadmap — Braintrust Feature Parity

This roadmap describes how to build the evaluation capabilities that Braintrust
offers as a managed platform, directly into this codebase. Each phase is
self-contained and adds a concrete, usable capability. Phases are ordered from
quickest win to most complex.

## Related Plans

- Playwright UI-E2E-Abdeckung (essentielle Interface-Flows, gemocktes API):
  `.cursor/plans/playwright_essential_e2e_975e1548.plan.md`

---

## Done — Run History & Dashboard

Every eval run records what it tested (git commit, models, `retriever_k`,
prompt fingerprints, metrics version, dataset fingerprint, label) in the
committed `eval/history/runs.jsonl`, and `python -m eval.dashboard` renders a
local HTML page with trends, run-vs-baseline comparisons, a per-topic
breakdown and a question-by-run grid. See `eval/README.md`.

This covers part of Phase 3's baseline idea: a run labelled `baseline` is what
later runs are compared against; `eval/data/baseline.json` can be derived from
the history instead of maintained separately. Phase 2's cost and latency can
be added to the run header when implemented. When Phase 1 lands, bump
`METRICS_VERSION` so the dashboard marks the scoring change.

## Done — Scoring v2 (supersedes Phase 1 below)

Metrics were redesigned after reading the RAG evaluation literature (notes in
the second brain): golden items carry nuggets (vital/okay facts) with verbatim
evidence quotes; an independent judge answers two narrow questions (nugget
assignment, groundedness); everything else is deterministic: evidence recall
in the first retrieval and in the context, strict and lenient vital-fact
recall, grounded recall, context utilization, `grade_documents` accuracy, and
a per-question error type that separates retrieval from generation failures.
Graded rather than binary scores come from nugget fractions instead of asking
the LLM for a number. Judge candidates are checked with `eval.judge_check`
against 16 calibration fixtures; saved runs can be re-scored without running
the graph. See `eval/README.md`.

### Next

1. **Golden set v2** (drafted, in review): 39 questions. The 15 new ones add
   7 English questions about Danish law, 2 pairs where the Danish rule and the
   IAEA value differ (apprentices' eye lens 15 vs 20 mSv; fetus 1 mSv vs "same
   protection as the public"), annex values and cross-references, and 4
   questions the sources do not answer (`expected_behavior: refuse`), one of
   them in-domain (the fee amount, which the sources mention but do not give).
   The judge is chosen: `qwen3.8-27b` on Scaleway (16/16 fixtures) with three
   groundedness votes.
2. **Scaleway by default** (done): evidence recall on the 24 reviewed questions (retrieval
   only, identical chunks, k=3): `bge-multilingual-gemma2` with query
   instruction 0.90, Gemini 0.81, `qwen3-embedding-8b` 0.76 (0.67 without
   instruction), BGE without instruction 0.25, Ollama `nomic-embed-text` 0.56.
   BGE vs Gemini: 3 questions better (the IAEA medical/occupational misses),
   1 worse — sign test p = 0.62, so a direction, not a proof. A full run with
   BGE embeddings passed 79 % vs 71 % with Gemini. Scaleway is now the default
   for answers and embeddings; Gemini stays as an optional provider and can be
   removed later if it goes unused.
3. **Retrieval experiments**, each a labelled run, cheapest first: more chunks
   per collection (`retriever_k`), BM25 fused with the dense retriever (RRF),
   the Danish translation of non-Danish questions as an extra query (RRF), and
   HyDE in Danish; a re-ranker only if evidence is retrieved but ranked too
   low.

## Retrieval Improvement — grounded in the eval (2026-09-29)

Ten retrieval and evaluation papers were read in full and checked against the
code (notes and ranking in the second brain:
`Concepts/Retrieval-Improvement-Grounded-in-Evaluation`). The rule: a retrieval
change is adopted only if it wins on the retrieval-only eval at an equal amount
of retrieved text, with the sign test on paired questions. Order, cheapest and
most enabling first:

1. **Eval first** — [#128](https://github.com/eikrad/Radiationsafety/issues/128):
   evidence recall at a fixed text budget (chunking changes chunk length), MRR
   (not implemented yet; needed for the re-ranking decision), and evidence
   position in the generator context. So far evidence recall predicts answers:
   19/21 questions with full evidence passed, 0/3 without.
2. **Fix the sufficiency grader's input** —
   [#129](https://github.com/eikrad/Radiationsafety/issues/129):
   `truncate.py` shows the grader 420 characters of each ~2500-character
   chunk. It caught 0 of 3 insufficient retrievals. Also: an IAEA value where
   a Danish rule applies counts as insufficient. Add labelled insufficient
   cases by removing evidence chunks. No hard abstention.
3. **Structure-aware Danish chunking** —
   [#130](https://github.com/eikrad/Radiationsafety/issues/130): keep
   `Paragraf`/`Stk` boundaries from the XML and prepend
   `law › Kapitel › §` to each chunk. Late chunking does not fit the
   API-hosted embedder, and semantic chunking gains little on real documents.
4. **BM25 + dense via RRF** —
   [#131](https://github.com/eikrad/Radiationsafety/issues/131): with a
   Danish/English stemming analyzer; RRF rather than tuned weights. Keep the
   Danish translation query: lexical matching fails across languages.
5. **Context order by jurisdiction** —
   [#132](https://github.com/eikrad/Radiationsafety/issues/132): Danish
   chunks currently sit at positions 4–6 behind IAEA; measure the position
   effect on the current generator.
6. **Grow the golden set with LLM-proposed, human-confirmed evidence** —
   [#133](https://github.com/eikrad/Radiationsafety/issues/133): check judge
   agreement on the 24 existing questions first (Danish and English
   separately); the judge's "not relevant" counts as unjudged.

Still deferred: re-ranking (only if MRR shows evidence ranked low), HyDE,
propositions / sentence-level indexing, eRAG as a routine metric (about 430
extra calls per run).

**Baseline E0 (2026-09-30, golden set v2, 39 questions, BGE embeddings,
`gemma-4` answers, `qwen3.8-27b` judge).**

- Retrieval is deterministic: two retrieval-only runs agree on every
  question, and the deep list's top 3 matches the graph's retrieval everywhere.
- Evidence recall @1 0.53, @3 0.88, @5 0.92, @10 0.94, @20 0.97; MRR 0.72.
  All 6 retrieval misses are Danish law, and 5 of them have the evidence at
  rank 4-13: a ranking problem more than a recall problem. Every IAEA
  question passes.
- The jurisdiction trap works: the English question on the Danish limits
  for 16-18-year-olds misses its evidence (rank 5).
- Pass rate 74 % and 77 % on identical runs: 3 of 39 questions flip
  between runs (all on "unsupported claim"), so differences under about
  3 questions in a full run are judge noise.
- `grade_documents` is right on 64-67 %: it flags 8-9 of 29 sufficient
  retrievals and passes 5 of 10 insufficient ones (#129).
- 13-15 % of answers show "could not be fully verified", with only local
  sources (#136). 3 of 4 refusals are right; the power-line question is
  answered anyway.
- Danish evidence sits at context position 4-5, behind the IAEA chunks
  (#132).

**Phase 3a (2026-09-30): graders read whole chunks.** On the same 39
questions:

- `grade_documents` right 0.64-0.67 → 0.85 (full run) and 0.82 (`--regrade` on
  the baseline's retrievals). Wrongly flagged sufficient retrievals dropped
  from 8-9 to 1 of 29.
- Still weak in the other direction: it passes 5 of 10 insufficient
  retrievals, and with the evidence chunks removed it still says
  "sufficient" for 15 of 29 (`grade_documents_ablation_correct` 0.48). Some
  of those may be valid unlabelled passages that restate the fact.
- Warnings on local-only answers 13-15 % → 0 % (#136). The trade-off: the
  one answer the judge flags for an unsupported claim now gets no warning;
  in the baseline the warning caught 1 such answer per run, along with 3-4
  false alarms.
- Pass rate 77 %, unchanged within judge noise.

**Step 1 (2026-10-02): one copy per Danish law.** The five orders were indexed
from both XML and PDF (same version); the PDFs are now skipped (Danish
collection 1321 → 803 chunks). Retrieval-only against the baseline: evidence
ranked higher for 8 questions and lower for 8 (sign test p = 1), recall@3
0.88 → 0.85, recall@20 unchanged at 0.97, still 6 retrieval misses. But they
are different questions: the three multi-nugget questions whose evidence
had to share 3 slots with duplicates now pass (area classification, dose
constraints, the 16-18-year-olds trap), while three single-fact questions
lost rank (registration rank 2 → 6, deregistration 1 → 4, fetus dose 2 → 4).
In each of those the short docling chunk of the PDF had ranked above the
2500-character XML chunk that holds the same sentence. Full run: 31/39 pass
(30 before, within noise), `grade_documents` right 0.90, one warning. The
duplicates were not only redundant: the PDF chunks had the better grain.
That is the case for structure-aware chunking of the XML (#130), next.

**Step 2: Danish law chunked along its structure (#130, 2026-10-02).** The
XML is cut along its own elements instead of every 2500 characters
(`ingestion_dk.py`): paragraphs of one group (same chapter, same group title)
are packed together up to 1500 characters, a longer one is split only between
its Stk., numbered items, lines or table rows, a split table repeats its
header row, and every chunk starts with `law (BEK nr, date) › chapter › group
› §`. No overlap: no cut falls inside an item. Five orders, 308 chunks
(median ~1300 characters, none above 1700); tests check that every word of
each law lands in a chunk and every golden quote found in a law lies within
one chunk. It replaces the character splitter rather than adding a parallel
index: only the Danish collection changes, `ingestion.py --dk-only` rebuilds
it in minutes and a revert restores the old chunks.
Adoption rule, set before measuring: against `dk-one-copy`, retrieval-only
evidence ranks improve for more questions than they worsen, at least two of
the three single-fact questions lost in step 1 (registration, deregistration,
fetus dose) are back in the top 3, none of the three gained in step 1
(area classification, dose constraints, 16-18-year-olds) is lost, and the full
run passes at least 30/39.
Result (2026-10-02, adopted, all four conditions met): evidence ranked higher
for 8 questions and lower for 4; recall@1 0.55 → 0.61, recall@3 0.85 → 0.90,
recall@5 0.90 → 0.97, recall@20 0.97 → 1.00, MRR 0.70 → 0.77. Deregistration
(4 → 1) and fetus dose (4 → 2) are back, registration only reaches 4 (from 6);
the step 1 gains held or improved (area classification 3 → 1, dose constraints
3 → 2), the 16-18-year-olds slipped 2 → 3 (still found). Full run: 31/39 as
before, retrieval misses 6 → 4, `grade_documents` right 0.90 → 0.92; the new
failures are one unsupported claim and a refusal question answered with a
warning, both in answers whose evidence did not change. Two questions lost
rank, and both are definitions: the definitions paragraph (§ 3 of BEK 1384,
about 100 numbered terms) is packed into 12 chunks of 5–7 terms, so one term
is a small part of its chunk (safety assessment 11 → 20, receipt inspection
3 → 4). Not pursued as its own step: a definitions-only chunk rule would be
designed on these two questions and judged on the same two (safety assessment
was a miss before too, receipt inspection moved one place). Both look up an
exact term, which lexical search (BM25 + RRF, #131) addresses for every
document; whether term lookups still fail is checked after that step, on
new definition questions written before any change.
The full run also exposed two grader replies the graph could not read (a
field given as an object, a verdict written as prose); both now get one
follow-up request for the JSON and, failing that, the cautious verdict.

**Step 3: dense + BM25, fused by reciprocal rank (#131, 2026-10-02).** Behind
`HYBRID_RETRIEVAL` (off by default). Per collection, the 50 best dense chunks
and the 50 best BM25 chunks are fused with RRF (k = 60, Cormack et al. 2009)
and the top k taken; the eval's deeper retrieval goes through the same path.
BM25 analyzes with the Snowball stemmer and stop words of the collection's
language (Danish law: Danish, IAEA: English), the condition under which BM25
matched learned sparse retrieval in BGE-M3's appendix C.2. Nothing is tuned:
k = 60 and 50 candidates are fixed before measuring. Expected limits: BM25
cannot match an English question against Danish text (49 vs 77 Recall@100 for
Danish→English in BGE-M3's MKQA table), so gains, if any, come from term
lookups within one language. Two evidence quotes that only matched the
removed PDF text are dropped from the golden set (no score changes; a new
retrieval-only reference run is taken on the cleaned set).
Adoption rule, set before measuring: retrieval-only with the switch on,
against the same run with it off, ranks the evidence higher for more questions
than lower, and neither recall@3 nor recall at the text budget is lower; the
full run with the switch on passes at least 30/39 (31/39 now). If it wins, it
is switched on by default in its own small PR.
Result (2026-10-02, rejected on all three conditions; the code was removed
again and stays in the git history, commit 03e9d51): the dense-only reference
reproduced step 2 exactly. With BM25 fused in, evidence ranked higher for 3
questions and lower for 13 (sign test p ≈ 0.02); recall@3 0.90 → 0.70,
recall at the text budget 0.97 → 0.83, MRR 0.77 → 0.63; full run 31/39 → 26/39
with 10 retrieval misses instead of 4. The losses were where predicted and
beyond: 6 of the 7 English questions about Danish law fell (BM25 has nothing
to match; the seventh did not move), but so did three English IAEA questions
(patient dose limits 2 → 12, transport index 1 → 6, lab spill 1 → 2). Of the
Danish questions three rose by one place and four fell: area classification
1 → 9, registration 4 → 6, and the two definitions BM25 was meant to help
(safety assessment 20 → not in the top 20, receipt inspection 4 → 8);
"sikkerhedsvurdering" occurs in many chunks, so the defined term is not a
rare one. The mechanism is RRF's equal weight:
BM25's first chunk scores 1/61 and outranks dense's second at 1/62, so a
weaker ranker pushes the stronger one's hits down. A weighted fusion would
need tuning on these 39 questions, which the plan rules out. Lexical search
is closed for this corpus and encoder; exact-term lookups stay open.

**Step 4: does the evidence's position in the context matter? (#132,
2026-10-02).** The context lists the 3 IAEA chunks before the 3 Danish ones,
so Danish evidence sits at position 4 of 6. Lost in the Middle found answers
worse with the evidence in the middle, but for 2023 models, 10-30 short
passages and short factoid answers; the effect shrank with stronger models and
was never measured at 6 long chunks. So it is measured first, with the
current generator: `eval.position_test` answers every answerable question
whose first retrieval holds vital evidence twice from the same chunks, the
evidence first and the evidence in the middle (after half of the other
chunks: position 4 of 6, where the best Danish chunk sits today). The judge
labels the vital nuggets of both answers (lenient vital recall); questions
where the two scores differ go into a sign test. About 6 calls per question.
Adoption rule, set before measuring: if the evidence first scores higher on
more questions than in the middle with p < 0.05, the context is reordered so
that each collection's best chunk comes first (Danish 1, IAEA 1, Danish 2, …:
Danish first because Danish rules apply in Denmark) and a full run confirms it
(at least 30/39). Interleaving rather than "best at both ends": only front
against middle is measured, the end is not. If p ≥ 0.05, the position effect
is not detectable here, the order stays, and k is the next question (step 5),
where distractors rather than position decide.
Result (2026-10-02, `position-k3`, not adopted: no position effect): 32
questions had vital evidence in their first retrieval (the three without it
are the known misses). The answer with the evidence first and the answer with
it at position 4 got the same vital-recall score on 31 of them; one Danish
question scored higher with the evidence first (radiography coordinator 0.50
vs 0.33). Mean lenient vital recall 0.98 first vs 0.97 in the middle, sign
test p = 1. The scores sit near the ceiling, so a small effect could hide
there, but none that would change answers at k = 3: once the evidence is
retrieved, this generator finds it at either position. The order stays.

**Step 5: k, the chunks per collection (2026-10-02).** At k = 3 the evidence
for 3 answerable questions sits at rank 4 or 5 (registration, receipt
inspection, radiography coordinator): recall@3 0.90, recall@5 0.97. More
chunks can bring that evidence in, but also distract: answer accuracy
saturates while recall keeps rising (Lost in the Middle §5), so k is judged by
answers, per group of questions (the rule in Concepts/Retrieval-Improvement-
Grounded-in-Evaluation). `eval.k_test` answers every golden question from the
top 3 and from the top 5 of each collection, the same retrieval otherwise, and
groups the questions: evidence at k=3, evidence only at k=5, evidence missing,
should refuse. The score is lenient vital recall, or for a question to refuse
1 for a refusal without unsupported claims. `RETRIEVER_K` (default 3) makes k
settable without a code change.
Adoption rule, set before measuring: k = 5 becomes the default, in its own
small PR confirmed by a full run passing at least 31/39, if (1) among the
questions with evidence only at k=5, k=5 scores higher on more of them than
lower; (2) over all other questions together, k=5 scores lower on at most 2
more questions than it scores higher (the position test saw 1 single-vote
difference in 32); and (3) at most 2 more answers carry an unsupported claim
at k=5 than at k=3. Otherwise k stays 3.
Result (2026-10-03, `k3-vs-k5`, adopted: all three conditions met): the 3
questions with evidence only at k=5 all scored higher at k=5 (registration
0.25 → 1.00, receipt inspection 0.25 → 1.00, radiography coordinator
0.33 → 1.00). Of the other 36, 34 tied, one scored higher at k=5 and one lower
(pre-disposal steps 1.00 → 0.75): no net loss. The 4 questions to refuse were
refused at both k. Answers with an unsupported claim: 2 at k=3, 1 at k=5.
The extra chunks did not distract this generator; they brought the evidence.
One oddity for the golden set: the safety-assessment definition was answered
fully at k=5 although its listed evidence (rank 20) is not in either context,
so another retrieved chunk states it, a pooling candidate (#133). It costs
about two thirds more context per call (10 chunks instead of 6) for the
generator and the graders.
Confirmation (2026-10-03, full run `k5`): 32/39 pass (31/39 at k = 3), so k = 5
is now the default (`RETRIEVER_K`). Retrieval misses 4 → 1, vital recall
0.89 → 0.98, `grade_documents` right 0.92 → 0.97. Registration and the
radiography coordinator now pass; receipt inspection has its evidence but
fails on an unsupported claim; pre-disposal steps misses a fact (as in the k
test), and area classification and the mobile X-ray question swapped
pass/fail on unsupported claims, within the judge noise of about 3 questions.

**Revised order (2026-09-30).** With evidence recall at 0.90 on 24 questions,
a retrieval change can fix at most 2–3 questions, too few flips for the sign
test. So the measurement comes first, then cheap and reversible changes, then
the costly ones:

1. **Eval harness** (#128, done): retrieval-only runs rank the evidence 20
   deep and derive recall@k, MRR and recall at a text budget from one list;
   compared by rank changes (more signal than recall@3 flips). Guards: index
   fingerprint, no recorded runs on uncommitted code, `golden --check-index`,
   pooling report for human-confirmed evidence (#133 without an LLM).
2. **Golden set v2** (39 questions, in review): English questions about
   Danish law, questions to refuse, harder questions (annex values,
   cross-references); then one
   baseline (retrieval-only twice to confirm determinism, full run twice).
3. **Graders see whole chunks** (#129, #136).
4. **BM25 + RRF** (#131, step 3: measured and rejected, see above); the Danish
   translation query as a second variant once the golden set has English
   questions about Danish law.
5. **Structure-aware chunking** (#130): planned as a parallel index; it
   replaces the character splitter instead (step 2 above), since only the
   Danish collection changes and `--dk-only` rebuilds it in minutes.
6. **Context order** measured by a position test (#132, step 4: no effect,
   order stays), then **k** (step 5)
   decided from the rank data plus a full run stratified by evidence presence.

Each step states its adoption rule before the measurement; parameters are not
tuned (RRF k = 60, fixed header format). A change that wins is switched on by
default in its own small PR.

## Generation — after retrieval (2026-10-10)

**The generation prompt (#145).** In the full run `k5` (32/39) the vital
evidence reached the generator for 34 of 35 answerable questions; 6 of the 7
failures happened after retrieval. The judge's reasons show what went wrong:
the three unsupported claims are legal conditions restated with a different
meaning ("samt efter indgreb" became "eller efter indgreb", "kan medvirke til"
became "nødvendigt", "anlægget" became "området", and in the same answer
GSR-7's "shall" became "bør"); pre-disposal steps left out "from generation up
to disposal"; the MRI question was answered from two web pages; the power-line
question was refused, but the refusal added a related figure misquoted (< 1 %
for 1 %). The prompt had a typo in its key rule ("Uso only information"),
said nothing about scope or jurisdiction, and nothing about facts on topic but
absent from the context.

The revised prompt (`graph/chains/generation.py`) adds numbered rules: only
the context; keep the meaning of requirements, conditions, definitions and
limits exact (how binding, how conditions combine, what or whom a rule
applies to, numbers, units, references); say whether a statement is Danish
or IAEA, an IAEA requirement is not a Danish rule; give a definition or list
asked for in full; name what the context does not answer and do not fill the
gap; non-ionising radiation is out of scope even if a web source answers it.
The rules name categories, not the words of the failed answers (no "samt",
no MRI, no power lines), so that the measurement on these same questions
does not reward a prompt written for them. Trust-Score found that prompting
alone does not stop models from answering from their own knowledge, and
GroUSE that over-refusal is a failure mode of its own; so the change is
measured, including for refusals of answerable questions.

`eval.prompt_test` replays the context each question's generator saw in the
`k5` run (web results included) and answers it with the previous and the
revised prompt; only the prompt differs. Both answers are scored as in a
full run, with 3 groundedness votes.
Adoption rule, set before measuring: the revised prompt is kept if, on the
39 replayed contexts, (1) fewer answers carry an unsupported claim; (2) on
lenient vital recall it scores lower on at most 1 more answerable question
than it scores higher; (3) all 4 questions to refuse are refused without an
unsupported claim (the two out-of-scope questions answered in `k5`
included); (4) no more answerable questions are refused although their
evidence was in the context (`wrong_refusal`); and then (5) a full run passes
at least 32/39. Otherwise the previous prompt returns with only the typo
fixed, and a scope check before retrieval (#150) is the next step for the
refusals. The caveat stays: the rules come from the failures of the same 39
questions, so a held-out check needs new golden questions (#133).
First measurement (2026-10-10, `prompt-145`, rule not met): passed 33 → 35;
lenient vital recall higher on 3 answerable questions (emergency exposure,
pre-disposal steps, safety-assessment definition), lower on none; no
`wrong_refusal` with either prompt. Conditions 1 and 3 failed: 3 answers with
an unsupported claim for each prompt, and 1 clean refusal of 4 (previous 2).
The revised prompt refused all 4 questions to refuse, but the judge flagged
the refusal itself in the three out-of-scope answers: "non-ionising radiation
is outside the scope of the provided documents" counted as contradicted,
because the context held web pages on MRI SAR, laser classes and power-line
fields. The judge exempts "the information is missing" from unsupported
claims, not "this is out of scope", so it would flag the expected answer in
`golden.json` too. On answerable questions the revised prompt had no
unsupported claim (previous 1). Decision: not adopted on this measurement.
The judge gets a narrow fix, checked on new calibration fixtures with a
different topic (not on these answers), and the saved answers of both prompts
are judged again under the same rule.
Judge fix (2026-10-11): statements that a question lies outside the scope of
the documents, and why, are no longer unsupported claims, and such an answer
counts as a refusal. Two fixtures on UV tanning beds check it: a scope refusal
despite an answering web source (failed 3 of 3 runs before, passes after) and
a scope refusal that adds a wrong standard number (must still be flagged);
all 18 fixtures pass twice with `qwen3.8-27b`.
Second measurement (2026-10-11, `prompt-145-rejudged`, the same answers judged
again, rule met on all four conditions): passed 34 → 38; answers with an
unsupported claim 1 → 0 (the previous prompt's power-line refusal with the
misquoted "< 1 %"); clean refusals 2 → 4 of 4; lenient vital recall higher on
2 answerable questions, lower on none; no `wrong_refusal`. The area
classification answer of the previous prompt was flagged in the first
judging and not in the second: judge noise, which the paired design puts on
both prompts alike.
Confirmation (2026-10-11, full run `prompt-145`): 38/39 pass (`k5`: 32/39),
vital recall 0.98 → 1.00; the one failure is the known retrieval miss
(safety-assessment definition). Part of the gain is the judge fix, which
`k5` did not have: under the fixed judge the previous prompt reached 34/39 on
the replayed contexts, so the prompt's own share is about 34 → 38, from one
full run. **Adopted**: the revised prompt is the default.

**Scope: the web search widens sources, not the topic (2026-10-11).** A
refusal "even if a web source in the context answers it" raises the
question what the web search is for. It fills gaps of the index within the
subject: IAEA publications that are not indexed, newer versions, other
Danish orders and SST guidance; there `verify_trusted` can check the answer
against iaea.org, retsinformation.dk and sst.dk. Non-ionising radiation
belongs to other authorities and standards (Arbejdstilsynet, ICNIRP, IEC),
which the system cannot verify against, and answering it would turn the RAG
into a general web search. So questions outside ionising radiation are
refused whatever the web search finds. The out-of-scope questions still
reach the web search before they are refused; a scope check before retrieval
(#150) would save those calls. Open for the golden set: the annual fee
question (`dk-fee-open-sources-amount`) is labelled "refuse" because the
corpus lacks the amount, but with the web search on, the amount in the fee
order on retsinformation.dk would be a good answer. Its label is defined
relative to the corpus, not to the web search.

## Privacy mode — local models (2026-10-11)

The first full privacy-mode run (`llama3.1:8b`, `nomic-embed-text`) first
exposed three faults, fixed in #160: Ollama cut every prompt to its own
4096-token window (a 7241-token prompt was read as 4095 tokens), privacy-mode
eval runs searched the web although the API keeps privacy mode off it, and
the local Danish collection still held the chunks from before step 1. A
rebuild while `llama3.1:8b` occupied the GPU then indexed the two guidance
PDFs as 4 chunks instead of 636: docling could not run and ingestion fell
back to pypdf without failing; with the GPU free the local index matches the
Scaleway one chunk for chunk.
The run itself (k = 4, `OLLAMA_NUM_CTX=8192`) was stopped after 35 of 39
judged answers: 8 passed, 22 failed on retrieval (18 of the 22 Danish
questions, 4 of 13 IAEA ones), against 1 retrieval miss in normal mode. The
embedder is the bottleneck: `nomic-embed-text` v1.5 is English-only, and its
model card says the text "must" carry a task prefix (`search_query:`,
`search_document:`), which the code never added.

**Step P1: a multilingual local embedder.** `bge-m3` (Ollama, 1.2 GB, more
than 100 languages, 8K context, no instruction for dense retrieval) is
compared with `nomic-embed-text` as it is today, retrieval-only, on the same
chunks: its collections are re-embedded from the Scaleway index
(`--reembed-from scaleway`), and local models now get collections of their
own (`-ollama-<model>`; the default keeps `-ollama`). Questions are embedded
on the CPU: next to `llama3.1:8b` on a 6 GB card, `bge-m3` on the GPU forced
a reload of 4-7 s per switch, on the CPU a question takes 0.3 s with the
same vector.
Adoption rule, set before measuring: `bge-m3` becomes the privacy-mode
default if, against `nomic-embed-text`, (1) it ranks the evidence higher for
more questions than lower (sign test reported), (2) recall@5 rises overall
and on the questions about Danish law (`source: dk-law`, either language),
and (3) a full privacy-mode run at k = 5 then passes more questions than the
stopped nomic run (8 of 35 judged). Otherwise `nomic-embed-text` with its
task prefixes is the next candidate. Privacy-mode users would have to build
the new collections (`ingestion.py`), which the "index not ready" message
already tells them.
Retrieval-only result (2026-10-11, `privacy-emb-nomic` vs
`privacy-emb-bge-m3`, conditions 1 and 2 met): evidence ranked higher for 25
questions and lower for 3 (sign test p < 0.001); recall@5 0.40 → 0.90, on
Danish law 0.18 → 0.89, IAEA 0.82 → 0.91; recall@1 0.17 → 0.47, MRR
0.28 → 0.66 (Scaleway's `bge-multilingual-gemma2`: recall@5 0.97).
The first two full runs for condition 3 did not finish. At k = 5 with the
default 16384-token window the kernel's OOM killer stopped Ollama after 20
answers: on the 13 GB test laptop `llama3.1:8b` held 6.2 GB of RAM next to
`bge-m3` on the CPU and the other programs. At k = 4 (8192 tokens) llama
replied `{}` to the sufficiency grader in `retrieve_missing`, which, unlike
the other graders, let the error end the request; it now counts a missing
verdict as insufficient, and Ollama, like Scaleway, is asked once more for
the JSON. The k = 5 run for condition 3 is repeated with a 12288-token
window: privacy-mode prompts reach about 4200 tokens before
`retrieve_missing`, and the smaller window keeps about half as much of the
model in RAM.
Confirmation (2026-10-11, full run `privacy-bge-m3-k5`, condition 3 met,
**adopted**): 20/39 pass (the stopped nomic run: 8 of 35); retrieval misses
22 → 4, evidence in the generator context 0.90. `bge-m3` is the privacy-mode
default; `OLLAMA_EMBED_MODEL=nomic-embed-text` keeps the old collections.
The failures are now the answer model's: 12 unsupported claims and 2
generator misses from `llama3.1:8b`. The run took 94 minutes: on the 13 GB
laptop `llama-server` held about 9.7 GB (5.0 in RAM, 4.7 swapped out), so
questions waited on swap; the next step is a smaller answer model that fits
the GPU.

**Step P2: a local answer model that fits the GPU.** `qwen3.5:4b` (Ollama,
3.3 GB, 201 languages) against `llama3.1:8b` (8 official languages, Danish
not among them), with everything else as in `privacy-bge-m3-k5`: `bge-m3`,
k = 5, a 12288-token window, the same judge. Local models now answer with
thinking off (`think=false`), or qwen3.5's `<think>` blocks would end up in
answers and grader JSON. The judge `qwen3.8-27b` is of the same family as
the candidate, so both runs are also re-scored by `mistral-medium-3.5-128b`
(neither Qwen nor Llama; it passed all 18 judge fixtures twice).
Adoption rule, set before measuring: `qwen3.5:4b` becomes the privacy-mode
default answer model if (1) its full run passes more questions than the
llama run (20/39), with per-question changes and a sign test reported, and
(2) under the `mistral-medium` judge it also passes more questions than
llama does under that judge. Minutes per run and the share of the model in
VRAM are reported alongside. `qwen3.5:9b` (6.6 GB) does not fit the 6 GB card
and would spill into RAM like llama; only its memory footprint is checked.
Result (2026-10-11, `privacy-qwen35-4b-k5`, rule not met): under
`qwen3.8-27b` qwen passed 23/39 against llama's 20/39 (8 questions better,
5 worse, sign test p = 0.58); under `mistral-medium` both passed 22/39 (6
better, 6 worse). Condition 2 failed: the lead appeared only under the judge
of the candidate's own family, and three questions are within run-to-run
noise anyway. Lenient vital recall was higher for qwen under both judges
(0.92 vs 0.80; 0.95 vs 0.89), unsupported claims equal or slightly more
(12 vs 12; 12 vs 10). What did differ: qwen3.5:4b runs fully on the GPU at a
12288-token window (3.0 GB; 68 tokens/s against llama's 23, 71 % on the GPU),
and the run took 38 minutes against 94. `qwen3.5:9b` holds 53 % of 6.0 GB on
the GPU and generates 13 tokens/s, slower than llama.
Decision: **qwen3.5:4b becomes the default for hardware reasons, not for
answer quality**, which the rule did not show to differ. Privacy mode targets
laptops; on this 13 GB one llama3.1:8b held about 9.7 GB of RAM and once
brought the OOM killer down on Ollama. `OLLAMA_MODEL=llama3.1:8b` still
selects llama. Open: the 12 unsupported claims, now the main failure in
privacy mode as well (the #145 prompt is already in place), and a second
qwen run to measure the noise.
Also measured: llama at k = 4 with an 8192-token window passed 22/39 in 64
minutes, against 20/39 in 94 minutes at k = 5 (12288), so k = 4 was not worse
and faster for a model that spills into RAM.

---

## Product — export the chat as a report (2026-10-13)

Not an evaluation step, but a gap seen while working on the eval: an answer
can only be read in the page. Whoever needs it for a safety assessment or an
inspection file copies the text by hand and loses the sources, the warning,
the web-search marker and the information on how the answer was produced.
Planned in #170: a button that exports the whole chat or a single answer,
built in the browser from the messages already on the page (so it also works
in privacy mode and needs no new request), as Markdown first and then a
print-ready page or PDF. Each report carries the questions and answers, the
sources, the warning, the web-search marker, the model and whether privacy
mode was on, and a fixed note that the text is AI-generated and has to be
checked against the original standards.
Open before building: whether `/query` can return the document version per
source (an answer is only reproducible together with it), and whether the
chunks carry page or section numbers for the source lines. It builds on the
conversation kept in the browser (#135) and waits for streaming (#149) only
in that finished answers are exported.

---

## Phase 1 — Continuous Scoring

**Goal:** Replace binary (0 or 1) metric scores with genuine 0.0–1.0 continuous
scores. Binary scoring throws away signal: a partially-correct answer scores the
same as a completely wrong one.

**Why it matters:** Regressions and improvements that don't cross the 0.5
threshold are invisible today. Continuous scores let you track slow drift and
measure the real impact of prompt or retrieval changes.

> **Note (v0.3.0):** `GradeGeneration` was refactored as part of the Reflexion
> implementation. The schema now has `passed: bool` + `missing_info: str`
> (replacing the old `grounded` + `answers_question` pair). Task 1 below should
> add `score: float` to the simplified schema, not the old one.

### Tasks

1. **Update grader prompts** (`graph/chains/generation_grader.py`,
   `graph/chains/context_sufficiency_grader.py`):
   - Add a `score: float` field (0.0–1.0) alongside `passed` in `GradeGeneration`
     and alongside `binary_score` in `GradeSufficiency`.
   - Update the prompt to instruct the LLM to assign a numeric confidence
     level, not just yes/no.
   - Example rubric for faithfulness: 1.0 = fully supported, 0.75 = mostly
     supported with minor gaps, 0.5 = borderline, 0.25 = mostly unsupported,
     0.0 = contradicted by context.
   - Keep `passed: bool` for the Reflexion routing logic (it reads `passed`,
     not the float score).

2. **Update `eval/metrics.py`** to read the numeric `score` field instead of
   casting `passed` to 1.0/0.0. Keep the boolean fallback for backwards
   compatibility when `score` is absent.

3. **Update `eval/run_eval.py`**:
   - Add standard deviation and median alongside mean in the summary.
   - Add a score histogram (bucketed 0.0–0.2, 0.2–0.4, etc.) to the Markdown
     report so you can see the distribution at a glance.

4. **Update pass/fail logic**:
   - Default threshold stays at 0.5, but now a score of 0.3 is more meaningful
     than a score of 0.0.
   - Add an optional `--warn-threshold` (e.g. 0.7) that marks items yellow in
     the report even when they pass.

**Estimated effort:** 1–2 days  
**Files touched:** `graph/chains/generation_grader.py`,
`graph/chains/context_sufficiency_grader.py`, `eval/metrics.py`,
`eval/run_eval.py`

---

## Phase 2 — Native Observability (Cost & Latency Tracking)

**Goal:** Record token counts, estimated cost, and per-node latency for every
eval run — stored alongside the existing JSON/Markdown reports. No external
service needed.

**Why it matters:** Right now you have no visibility into how expensive a run
is, which retrieval or generation step is slow, or how cost/latency trends over
time. This phase closes that gap without adding LangSmith as a dependency.

### Tasks

1. **Add a timing decorator / context manager** (`graph/tracing.py`, new file):
   - Wrap each LangGraph node with a lightweight timer that records wall-clock
     duration in milliseconds.
   - Store results in a `run_metadata` dict that travels through `GraphState`.

2. **Add token counting** to each LLM call site:
   - LangChain's `ChatModel` responses include `response_metadata.usage` (input
     tokens, output tokens). Extract these after each call.
   - Accumulate totals in `run_metadata`.

3. **Add a cost estimator** (`eval/cost.py`, new file):
   - Maintain a small dict of `{provider: {model: (input_$/1k, output_$/1k)}}`.
   - Compute estimated cost from token counts. Mark as "estimated" in the
     report since pricing changes.

4. **Extend the eval report schema** (`eval/run_eval.py`):
   - Per-question: add `latency_ms`, `input_tokens`, `output_tokens`,
     `estimated_cost_usd`.
   - Summary: add total cost, mean latency, slowest question, highest-cost
     question.

5. **Print a cost/latency table** in the Markdown report after the per-question
   breakdown.

**Estimated effort:** 3–5 days  
**Files touched:** `graph/state.py`, `graph/graph.py`, `graph/nodes/*.py`,
`eval/run_eval.py`, new `graph/tracing.py`, new `eval/cost.py`

---

## Phase 3 — CI/CD Regression Gating

**Goal:** Run eval automatically on every pull request via GitHub Actions and
block merges when scores regress below the current baseline.

**Why it matters:** Right now, a PR that quietly breaks faithfulness from 0.85
to 0.60 will be merged with no warning. This phase makes metric regressions
visible on every PR before merge.

### Tasks

1. **Create a baseline file** (`eval/data/baseline.json`):
   - Schema: `{"faithfulness": float, "answer_relevance": float,
     "context_precision": float, "context_recall": float, "recorded_at":
     ISO-date}`.
   - Commit an initial baseline generated from the current golden set on
     `master`.
   - Add a CLI subcommand `eval.run_eval --update-baseline` that overwrites
     this file and prints a diff.

2. **Write a comparison script** (`eval/compare_baseline.py`):
   - Reads the latest report JSON and `baseline.json`.
   - For each metric, computes delta and flags as regression if delta < `-0.05`
     (configurable via `--tolerance`).
   - Exits with code 1 on regression, 0 on pass. Prints a human-readable diff
     table.

3. **Add a GitHub Actions workflow** (`.github/workflows/eval.yml`):
   ```
   Trigger: pull_request (paths: graph/**, eval/**, prompts/**)
   Steps:
     1. Checkout + uv sync
     2. Run ingestion (restore from cache if unchanged)
     3. uv run python -m eval.run_eval --limit 20 --no-web-search
     4. uv run python -m eval.compare_baseline
     5. Upload report artifact
   ```
   - Use `--limit 20` for PR checks (fast) and full 80-item set on merge to
     `master`.
   - Cache the `.chroma` vector store by hashing `documents/` so ingestion only
     runs when documents change.

4. **Post a PR comment** with the metric diff table using the GitHub Actions
   built-in token (`GITHUB_TOKEN`). No external service needed.

5. **Update baseline on merge**: add a step in `ci.yml` (the existing workflow)
   that runs `--update-baseline` after a successful merge to `master` and
   commits the updated `baseline.json`.

**Estimated effort:** 2–3 days  
**Files touched:** new `eval/data/baseline.json`, new `eval/compare_baseline.py`,
new `.github/workflows/eval.yml`, existing `.github/workflows/ci.yml`

---

## Phase 4 — Production Trace Capture & Dataset Growth

**Goal:** Log every real production query with its full context and response to
a local JSONL file. Provide a CLI command to promote any logged trace to the
golden dataset with one command.

**Why it matters:** The golden set today has 80 hand-crafted questions. Real
user queries surface edge cases and failure modes that synthetic data misses.
This phase lets the golden set grow from production usage.

### Tasks

1. **Add a trace logger** (`api/trace_log.py`, new file):
   - Append one JSON line per query to `eval/data/traces.jsonl` (gitignored by
     default).
   - Each line: `{id, timestamp, question, generation, context_used,
     retrieval_warning, web_search_attempted, latency_ms}`.
   - Wire it into `api/main.py` after the graph returns a result.
   - Make logging opt-in via `TRACE_LOG_ENABLED=true` in `.env` (off by
     default for privacy).

2. **Add a thumbs-down endpoint** (`api/main.py`):
   - `POST /api/feedback` with body `{trace_id: str, rating: "bad" | "good",
     comment: str | null}`.
   - Appends a feedback record to `eval/data/feedback.jsonl`.
   - Wire a thumbs-down button in the frontend chat UI.

3. **Add a `promote` CLI command** (`eval/promote.py`, new file):
   - `uv run python -m eval.promote --trace-id <id>` reads the trace from
     `traces.jsonl`, formats it as a golden item, and appends it to
     `eval/data/golden.json`.
   - Interactive mode: `uv run python -m eval.promote --interactive` lists
     recent traces (newest first, bad-rated first) and lets you select, review,
     add `expected_answer` and `key_facts`, then append.
   - Deduplicate by question text (warn if similar question already exists).

4. **Add a `list-traces` CLI command** that prints a table of recent traces
   with columns: `id | date | question (truncated) | rating | promoted`.

**Estimated effort:** 4–6 days  
**Files touched:** new `api/trace_log.py`, `api/main.py`, new
`eval/promote.py`, `frontend/` (thumbs-down button), `eval/data/.gitignore`

---

## Phase 5 — Prompt Experimentation

**Goal:** Run eval against two different prompt or retrieval configurations
side-by-side and generate a comparison report, so you can measure the real
impact of a prompt change before committing it.

**Why it matters:** Right now you can only compare prompts by running eval
twice, manually diffing the Markdown reports, and trying to remember what
changed. This phase makes A/B comparison first-class.

### Tasks

1. **Extract prompts into versioned config files** (`prompts/`, new directory):
   - Move the system prompts from `graph/chains/generation.py`,
     `graph/chains/generation_grader.py`, etc. into `.txt` or `.yaml` files
     under `prompts/`.
   - Each chain reads its prompt from the config file at startup (path
     configurable via env var or constructor arg).
   - Add a `prompts/default/` variant that mirrors the current prompts exactly,
     so nothing changes in behaviour until you create a new variant.

2. **Add `--prompt-variant` flag** to `eval/run_eval.py`:
   - `--prompt-variant prompts/experiment-1/` loads prompts from that directory
     instead of the default.
   - The eval report records which variant was used.

3. **Write a comparison script** (`eval/compare_variants.py`):
   - Takes two report JSON files as arguments.
   - Outputs a Markdown table: per-metric delta (A vs B), per-question winner,
     and overall recommendation.
   - Example: `uv run python -m eval.compare_variants report_A.json report_B.json`

4. **Add a convenience shell script** (`scripts/experiment.sh`):
   - Runs eval twice (once with `--prompt-variant A`, once with `B`) and then
     calls `compare_variants.py` automatically.

**Estimated effort:** 3–4 days  
**Files touched:** new `prompts/` directory, `graph/chains/*.py`,
`eval/run_eval.py`, new `eval/compare_variants.py`, new
`scripts/experiment.sh`

---

## Phase 6 — Human Review Queue

**Goal:** Provide a simple terminal UI for domain experts (radiation safety
specialists) to review flagged answers, approve or correct them, and optionally
write corrections back to the golden dataset.

**Why it matters:** LLM-as-judge metrics can miss domain-specific errors. A
medical or regulatory expert can catch issues that faithfulness and relevance
scores miss. This phase adds a lightweight human-in-the-loop layer without
requiring a full web UI.

### Tasks

1. **Auto-flag items in eval reports**:
   - Mark a question as `flagged: true` in the JSON report when any metric
     score is below a configurable `--flag-threshold` (default 0.6, higher than
     the 0.5 pass threshold).
   - Add a "Flagged for review" section to the Markdown report.

2. **Write a review CLI** (`eval/review.py`, new file):
   - `uv run python -m eval.review --report eval/reports/report_<ts>.json`
   - Iterates through flagged items one at a time in the terminal.
   - For each item, displays: question, retrieved context (truncated), generated
     answer, metric scores.
   - Prompts the reviewer: `[A]pprove / [R]eject / [C]orrect / [S]kip`
   - Records decisions to `eval/data/reviews.jsonl`.

3. **Support corrections**:
   - `[C]orrect` opens `$EDITOR` (or a simple inline prompt) for the reviewer
     to type the correct answer and optional key facts.
   - Corrections are saved to `reviews.jsonl` and can be promoted to
     `golden.json` via `eval.promote --from-reviews`.

4. **Review summary report**:
   - `uv run python -m eval.review --summary` prints a table of all reviewed
     items: question, decision, reviewer comment, date.
   - Useful for tracking how often the LLM metrics agree with human judgment
     (calibration check).

5. **Optional — web UI** (future, not in this phase):
   - If the terminal workflow proves insufficient, the same `reviews.jsonl`
     format can back a simple FastAPI + htmx review page added to the existing
     backend. The data format is designed for this upgrade path.

**Estimated effort:** 4–6 days  
**Files touched:** `eval/run_eval.py`, new `eval/review.py`, new
`eval/data/reviews.jsonl` schema doc

---

## Implementation Order

```
Phase 1 — Continuous Scoring          (1–2 days)   ← start here, biggest signal/effort ratio
Phase 3 — CI/CD Regression Gating    (2–3 days)   ← protects against regressions early
Phase 2 — Observability               (3–5 days)
Phase 5 — Prompt Experimentation      (3–4 days)
Phase 4 — Trace Capture               (4–6 days)
Phase 6 — Human Review Queue          (4–6 days)   ← most complex, least urgent
```

Phases 1 and 3 are independent and can be worked on in parallel.
Phases 4 and 6 share the `reviews.jsonl` / `golden.json` pipeline and should
be developed together or in sequence.

---

## Feature Parity Summary

| Braintrust Feature              | Phase | Status  |
|---------------------------------|-------|---------|
| Continuous 0–1 scoring          | 1     | done (scoring v2: nugget fractions) |
| Score distributions in reports  | 1     | planned |
| Cost tracking per run           | 2     | planned |
| Latency tracking per node       | 2     | planned |
| CI/CD eval on every PR          | 3     | planned |
| Regression gating (block merge) | 3     | planned |
| PR comment with metric diff     | 3     | planned |
| Production trace logging        | 4     | planned |
| One-command dataset promotion   | 4     | planned |
| User feedback (thumbs down)     | 4     | planned |
| Prompt versioning               | 5     | planned |
| Side-by-side prompt comparison  | 5     | planned |
| Human review queue              | 6     | planned |
| Corrections → golden dataset    | 6     | planned |
| Experiment history & comparison| —     | done    |
| Per-topic score breakdown       | —     | done    |
| Retrieval vs generation errors  | —     | done    |
| Judge calibration unit tests    | —     | done    |
