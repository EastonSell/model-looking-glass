# Pair/dose saved-evidence audit

This CPU-only reader checks one quiescent, completed public copying attempt.
It hashes and bounds the files, checks the reported source, case, exposure,
checkpoint, runtime, process and scoped custodian records against separately
supplied pins, reconciles the journal, and independently replays all 576
records and 32 full-vocabulary files. Invalid, missing or partial evidence
fails; the reader does not repair, retry or discard an attempt.

The pre-run record says the independent reader was under review. The caller
must supply the **later** reviewed reader commit and SHA-256 values for the
three reader files in `reference_revision` and `reference_source_sha256`.
The reader compares the executing checkout and those Git blobs to the supplied
values. This audit-time binding does not assert that the reader was published
or preregistered before the study began. Independently establish the source
and publication of every pin; do not simply copy unverified values from the
attempt. Custodian and process records are reported observations, not
authentication, proof of descendant lifetime, or trusted timestamps.

With a reviewed checkout containing the later reader commit, construct a
JSON object containing every field in `EXPECTED_FIELDS` in
`looking_glass/copy_pairs_dose_evidence.py`, including the complete source
and asset hash maps and exact launch, context and receipt hashes. Then run:

```text
python -B scripts/audit_copy_pairs_dose_v1.py --attempt <quiescent-attempt> --cases <frozen-public-cases.json> --pins <independent-pins.json> --output <new-report-outside-attempt.json>
```

The reader rejects extra attempt files, reparse-point files, incomplete
receipts and inconsistent metadata. The caller must keep the checkout, Git
object store and attempt quiescent; path checks do not guard against a
concurrent mutator. The report says `saved_evidence_consistent` only for the
bytes and reported observations it checked. It cannot prove model execution,
physical cleanup, publication chronology, private disjointness, unknown
history or behavior outside these public, nonblind examples.
