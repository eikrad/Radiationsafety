# Radiation Safety RAG

Answering terms are defined in [CONTEXT.md](CONTEXT.md).

## Answering

**Context sufficiency**:
Whether the retrieved context contains every fact needed to answer a Question
under the applicable jurisdiction. An insufficient verdict identifies what is
needed and what was found; an absent verdict supplies no new assessment.
_Avoid_: Document relevance

## Documents

**Uploaded document**:
A PDF supplied through the Documents panel. It is registered as a source only
after its content has been indexed successfully with at least one chunk.
_Avoid_: Uploaded source

## Evaluation

**Saved evaluation run**:
An evaluation whose answers have been retained for judging and scoring again
without repeating retrieval or generation. Saving answers does not imply that
the evaluation completed.

**Recorded evaluation run**:
A completed evaluation included in the run history for comparison with other
evaluations.

**Rescore**:
A new assessment of saved answers to the same original questions, using current
grading criteria, judge, and metrics.

**Run provenance**:
The original code revision, recorded configuration, and exact questions that
identify what an evaluation measured.
