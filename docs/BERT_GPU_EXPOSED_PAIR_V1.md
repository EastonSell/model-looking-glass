# DistilBERT exposed-pair GPU replication v1

This is a fixed, public, nonblind **one-pair cross-runtime observation**. It
uses two rows already evaluated in the [balanced CPU study](https://github.com/EastonSell/model-looking-glass/blob/9d0aa9c0999924dd36b12bbdb638a27476d86bc8/results/bert-agreement-cpu-evaluation-v1-20261029/README.md).
It is not a new held-out evaluation, a lexical-control experiment, or evidence
of a uniquely localized grammatical computation. Unknown pretraining and
uncited prior exposure remain unknown.

Use the pinned `distilbert/distilbert-base-uncased` masked LM revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, evaluation mode, float32 eager
attention and a single owned CUDA device. The target is the first evaluation
family's singular/singular row `The painter beside the artist [MASK] ready.`
with IDs `[101,1996,5276,3875,1996,3063,103,3201,1012,102]`. The donor is
`The painters beside the artist [MASK] ready.` with IDs
`[101,1996,12499,3875,1996,3063,103,3201,1012,102]`. Both have ten ones in
their attention mask. The fixed output IDs are `is=2003` and `are=2024`.
Tokenization and the published manifest must be checked before model work.

At block 2, tuple output element 0, invocation 0, batch 0 and subject token 2,
execute four forwards in order: capture the target, capture the donor,
capture and self-patch the target from the same forward, then capture and
patch the target with the donor vector. Require exact raw float32 bit landing,
target pre-action consistency, finite full logits and exact mask-position
vocabulary bytes for the target/self control. Save the raw selected activation
vectors, the selected output bits and a SHA-256 of each full mask vocabulary
vector. Full mask logits cannot be reconstructed from their hashes.

Let `L = logit(is) - logit(are)` computed from stored float32 bits decoded
exactly into binary64. Report target, donor, self and patched margins, and
`L_target - L_cross` as an oriented shift toward the plural output: positive
means the `is − are` margin decreased. This sign alone does not guarantee the
patched margin is closer to the donor's margin. The previous CPU
study recorded approximately `+5.2040143`, `-4.2662587`, `+5.2040143`, and
`+0.0572338` respectively, giving movement `+5.1467805`; the patched CPU value
was close to a preference tie. Report a sign change if observed but do not
presume one, select a different pair after observing the output, equate this
single A-opposite effect to the earlier A/B-averaged statistic, or require
CPU and GPU bit identity. Record actual seeds, thread counts, numerical flags,
runtime versions, checkpoint/source hashes and the owner exit. The fixed GPU
owner seed is 17; the earlier CPU evaluation used seed 2026102810. The model
is evaluated with dropout disabled in both, but their results are not asserted
to be bit-identical.

Run only under the reviewed single-use supervised device owner and common
physical-device lease. Bind the checkout to the canonical workspace; check
fresh device free memory and headroom at admission and at synchronized stages;
apply and read back the native 25% allocator cap, retain the declared whole
owner and trace commitments through cleanup, and quarantine on uncertain
ownership or failed synchronization. Record cumulative PyTorch allocator
peaks, point-sampled device free memory and cleanup; these are not total GPU
memory maxima and cannot prevent external pressure between samples.
Partial stage records from incomplete outcomes are typed diagnostics and may
contain counters that did not pass the admission checks.
The owner checks the fixed PyTorch and CUDA build versions before its first
CUDA call, and the fixed device name and capability before allocator/model
work. The separate free and reserved counter reads are sequential; a
sample whose free plus reserved bytes exceeds the device total is rejected.
Lease and claim checks are sequential and rely on the dedicated cooperative
owner; they are not atomic against unrelated state changes. A failed drain
quarantines the owner and does not trigger another CUDA operation.
The owner retains its named input, output, source and capture objects once
they have been assigned to custody. An interruption between a native return
and that assignment can leave the returned object unrooted; the owner then
quarantines and exits. Model-internal intermediates rely on the trusted
PyTorch runtime and allocator and are not all individually rooted by the owner.
The supervisor tracks and reaps the direct owner; this run assumes that the
owner does not spawn CUDA-using descendants and does not independently prove
process-tree quiescence. A nonzero or unknown owner exit retains the unsafe
marker even if a clean attestation was written.
The direct owner exits without Python shutdown hooks after success or failure;
operating-system device-context teardown is outside that observation. Startup
source hashes assume a trusted, quiescent checkout for the whole run; they do
not attest bytes that would have executed after an in-run file change.
The dependency directory and fresh bytecode prefix also rely on trusted,
quiescent local state. Runtime versions are recorded; interpreter and native
dependency bytes are not independently attested.

The earlier four-forward, same-shape development calibration observed a
cumulative PyTorch allocated/reserved peak of 307,767,808/339,738,624 bytes.
That observation supports sizing but is not a worst-case guarantee or an
authorization for the changed input/workload. An independently reviewed exact
source and a separately completed recovery/slot release are required before
this attempt. Preserve incomplete attempts and their exposed inputs; no silent
retry or outcome-based changes are permitted.
