# Public cases for the matched-magnitude BERT study

These cases were assembled after the independently reviewed analysis was
published at commit `168939c651221e52bcc3a07bbe37c267ced49022`, whose protocol
SHA-256 is `91683a8f58ec182428f54c29e00d2bace692de3288014f3bf5f16bcbf665d35c`.
The case manifest SHA-256 is
`5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb`.

Without reading new model outputs, the author listed a convenience shortlist
of human-role nouns in `looking_glass/bert_matched_cases.py`. The selection
removes all prior cited BERT lemmas, retains words whose singular and ordinary
`s` plural are both separate entries in the pinned vocabulary, sorts the 18
eligible words lexically and takes the first 16. The first eight become
subject lemmas A; the last eight, in order, become distractors. The resulting
ordered pairs are architect/mechanic, captain/minister, designer/musician,
detective/officer, engineer/photographer, instructor/physician,
journalist/politician and judge/professor. Each family's B donor is the next
family's A subject, cyclically.

The manifest contains four subject-number by distractor-number targets and
eight distinct A/B rows per family: 32 targets and 64 rows total. The builder
compares every full token row with the 96 rows in the earlier public manifest,
and verifies the immutable v5 cited-exposure snapshot; no exact row or selected
lemma overlaps. The offline verifier checks pinned vocabulary/tokenizer hashes,
all 32 isolated singular/plural forms and all 64 full prompt IDs and masks.
The execution owner must repeat the full-row tokenizer check before its first
forward. This inventory is public, nonblind and selected for convenience;
unknown uncited history and pretraining are not certified. No model forward is
needed to assemble or verify it, and the manifest alone does not authorize a
study run.
