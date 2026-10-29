# Matched-magnitude saved-evidence audit

`looking_glass.bert_matched_evidence` is an independent, standard-library-only
consistency check for the fixed [matched-magnitude protocol](BERT_MATCHED_MAGNITUDE_V1.md).
It does not load a checkpoint, execute a model, select cases, or authorize a
forward. Its caller must bind the exact reviewed source, manifest, closed journal,
runtime and model identity, retain all original records, and establish that the
input is quiescent while it is read. The checker snapshots supplied Python
objects; it is not a sandbox against a concurrent writer.
The caller must independently validate the complete frozen manifest, including
globally distinct rows and the prescribed cyclic relationship between each
family's B lemma and the next family's A lemma and token IDs. This checker
validates each family's internal donor mapping and the supplied family order;
it does not establish those cross-family manifest conditions.

`audit_family_events(manifest, records, split=..., family=..., mask_id=...)`
requires one complete ordered family: 60 started/completed forwards, four dose
plans, and 20 successful controls. Missing, failed, extra, misordered or
inconsistent records are rejected. It checks input and donor identities, token
and site coordinates, ordered hook metadata, clean/self/final controls and
float32 vector bits. It independently reconstructs the six distances, shared
radius, integer nearest-even float32 rounding, artificial direction and landed
norms from the captured bytes. It verifies the saved dose plan against those
reconstructions and returns the per-case raw selected-logit bits, margins,
radius, distances, interpolation fractions, realized norms, errors, base
norms, and ratios.

`aggregate_families(families)` accepts eight previously audited family results
in order and recomputes the fixed case, family and cross-family analysis. The
primary effect is the signed B/opposite minus B/same contrast at matched dose;
the paired-site mean is reduced from per-case differences. A/same is a zero-dose
self-patch and is used only in the labeled secondary contrast. Artificial shifts
are descriptive. Baseline accuracy includes every case and treats ties as
incorrect. A sign reversal means that clean and opposite-donor margins have
strictly opposite signs, including incorrect-to-correct transitions; directional
counts are also reported, and ties are excluded. The returned cell summaries,
family means, 0.1-logit decision thresholds and six-positive-family criterion
are descriptive, not significance tests. The full vectors and ordered controls
remain in the original journal, not in the returned summary.

`audit_event_journal(manifest, path, expected_sha256=..., split='evaluation',
family_ids=(1, 2, 3, 4, 5, 6, 7, 8), mask_id=103)` checks the exact bytes of a
closed JSONL file, rejecting duplicate keys, nonfinite JSON values, bad UTF-8,
missing/reordered families, lines over 64 KiB, or files over 16 MiB. It processes
one family of events at a time and compares the file hash against the caller's
independently retained expected digest. Its result contains `schema`, `sha256`,
`size_bytes`, `record_count`, `forward_count`, `family_ids`, `families`, and
`aggregate`. A single-family rehearsal returns `aggregate: null`; it is not an
eight-family decision. The caller must still establish exclusive quiescent
custody: metadata and digest checks do not prevent an adversarial concurrent
writer, authenticate the expected hash, or prove prior publication.

The full mask logits are saved only as SHA-256 digests. Their equality can be
checked across the declared controls, but a digest cannot reconstruct the
underlying logits or independently prove that the two reported selected bits
came from that vector. The saved action metadata and tensors are recorded
evidence, not proof that hooks actually executed, state was restored, sources
were authentic, the manifest preceded execution, or data were previously
unseen. Source/execution custody and exposure accounting remain separate
responsibilities. The aggregate also does not authenticate its input family
summaries: the caller must first audit and bind every corresponding journal.
The 16 MiB input guard checks serialized family data after copying; it is not
a peak-memory or filesystem quota.

Run the fabricated-only checks with the test dependencies installed:

```text
python -m pytest -q tests/test_bert_matched_independent.py tests/test_bert_matched_saved_audit.py tests/test_bert_matched_saved_journal.py tests/test_bert_matched_runner.py
```

The tests use fabricated observer data and may also exercise an untrained CPU
model. No pretrained checkpoint or private study inputs are used by these checks.
