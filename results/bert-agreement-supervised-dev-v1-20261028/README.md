# Supervised DistilBERT agreement development retry

This CPU-only run repeated the eight already-exposed development-family-1
rows; no evaluation row was forwarded. Source commit
`b0dde0dd66712771a3a5b314dd4f2431c73f852c` was published before the recorded
local launch. The selected checkpoint, inputs, sites, donor pairs and scoring
are described in `docs/BERT_AGREEMENT_V1.md`. The immutable v2 exposure
snapshot records the earlier cited attempts; this additional attempt is a
subsequent exposure record, not a change to that snapshot.

The controller reported **complete** for run
`bb4d5c3acc37d606b527d1b1fd6b46b1`. The owner was recorded as reaped with exit
code 0, and the separate local wrapper observed controller exit code 0. The
supervisor recorded clean cooperative CPU-study custody. Its generic raw
`clean device teardown` reason is scoped to that CPU custody and does not
attest GPU state. The CPU unsafe marker was absent on post-run inspection.
These are cooperative records and local observations, not independent process
attestation or an independently witnessed publication timestamp.

The 140-forward journal has SHA-256
`ff14574bf1b35058b3cff51e4752180ec589982cf3a547baba318a9591af8b67`
and is byte-identical to the earlier saved development journals. To avoid a
duplicate copy, `inventory.json` identifies the existing compressed journal at
`results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz`. An
independent audit checked the source and artifact hashes, custody identities,
all intervention effects and 48 controls. Full mask logits were not retained:
their hashes cannot reconstruct them or attest faithful execution.

The four-case mean signed effects were 5.76077 at the candidate block-2
subject site, -0.22613 at the block-2 distractor site, 8.51654 at the fixed
block-0 subject comparison, -0.10013 at the block-3 distractor comparison and
-0.01488 at the block-4 distractor comparison. The block-0 subject comparison
exceeded the candidate effect. For the plural-subject/plural-distractor case,
both opposite-number patches left the `is` minus `are` margin negative, so they
did not flip that prediction. All four unpatched targets in this one development
family preferred the matching token. These are descriptive, exploratory
observations; they do not isolate a unique grammatical-number computation or
estimate generalization.

The copied JSON records retain their original bytes. The published wrapper is
explicitly redacted only at its absolute interpreter path; `inventory.json`
records the original local wrapper's SHA-256 and the redacted copy's hash.
The local exit record is a separate local observation. The earlier incomplete
attempt remains incomplete, and this retry does not change its status.
