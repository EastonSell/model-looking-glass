# Invariant-noun tokenizer feasibility

Before constructing full prospective cases, the pinned DistilGPT-2 tokenizer
was used on the candidate noun fragments and the eight specified continuations.
The [record](tokenizer-feasibility.json) reports the token IDs, alignment and
candidate lists. It contains no model output from the new follow-up. The
previous study's 48/48 baseline and 324/324 valid delivery observations are
included only as evidence that its previously used inputs and intervention
were executable; they do not establish model accuracy on `this/these` nouns.

The fixed [protocol](../../docs/AGREEMENT_INVARIANT_NOUN_V1.md) is published
before case construction. The candidate words were tokenizer-inspected before
that publication. Discovery and evaluation inputs and their results will be
public and nonblind; noun and frame partitions are held out only within this
new follow-up.

To reproduce this tokenizer-only record, provide the exact pinned tokenizer
JSON and the previous study's independently saved `science-replay.json` as
arguments to `tokenizer_feasibility.py`, with tokenizers 0.22.2 installed.
The helper checks their SHA-256s, imports no model and writes a new output
exclusively; it does not establish actual publication, exposure currentness,
model competence or execution authorization.
