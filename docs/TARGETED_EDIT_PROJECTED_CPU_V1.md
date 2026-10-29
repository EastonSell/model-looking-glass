# Projected rank-one edit: prospective CPU protocol v1

**Status: protocol only.** No cases for this follow-on have been assembled and
no model execution for it has occurred. Publish this protocol before assembling
its evaluation cases or running its discovery and calibration forwards. Freeze
the exact public cases, token IDs, continuation IDs, source revision and
checkpoint hashes in a separate manifest before any of those forwards. Record
that publication ordering as an operator attestation, not a trusted timestamp.
The preceding v1 negative result and the separate v2 dose-grid study are not
reclassified by this experiment.

This small experiment asks whether constraining an edit on a few observed
activations changes its measured target/locality tradeoff. It is not a claim
that a trace explains model reasoning, that a factual belief was edited, or
that capabilities are generally preserved. Target and evaluation inputs are
public and non-blind; they are held out from constructing the edit, not from
pretraining. Unknown prior exposure remains unverified.

## Model and fixed development inputs

Use `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, offline, eager attention,
float32, no cache, evaluation mode, CPU, one intra-op and one inter-op thread.
Record the full checkpoint-asset hashes, runtime/build identity, deterministic
and denormal settings, source hashes, input IDs, and the exact original state
and weight digests. Require the original state and site weight to match the
published v1 reference; do not save raw model weights. This is a cooperative
CPU execution contract, not a claim about other GPU workloads.
Also record and verify a digest of every named buffer, including nonpersistent
buffers, before and after every condition. The historical v1 state-dict
digest did not attest those buffers. In the study's own process call
`torch.set_flush_denormal(False)` and require success; probe float32
subnormal multiplication, normal halving and NumPy float32-to-float64
conversion before and after each weight addition and score calculation.
These calling-thread checks do not establish every library thread's mode.

The one fitting input is the existing public v1 discovery row, `The capital
city of France is`. Maximize the final-position margin
`m = logit( London) - logit( Paris)` at the single site
`transformer.h.5.mlp.c_proj.weight`, shape `[3072, 768]`. The four fixed
calibration rows are the existing v1 locality-5 through locality-8 inputs:
`The largest ocean is the`, `Two plus two equals`, `The opposite of hot is`,
and `A week has seven`. They are development data, not evaluation cases.
Capture the original-model `c_proj` input at **every token position** of all
four rows, using the exact published v1 IDs; concatenate them in that order
without centering. The expected matrix is `X` with 18 rows and 3072 columns.
The single discovery activation at the final position is `h` in `R^3072`.

Obtain `g` in `R^768`, the gradient of the discovery margin with respect to
the corresponding projection output. Independently compute the full weight
gradient `G` and require finite nonzero values and
`||G - h g^T||_F / ||G||_F <= 1e-5` in float64. Abort the attempt if the
shape, call count, residual or restoration checks fail; do not assume the
outer-product identity at other sites or layers.

## Frozen projection and controls

Promote the saved float32 `X`, `h` and `g` to float64 and require their norms
and `||W||_F` to be finite and nonzero. Compute `torch.linalg.svd(X,
full_matrices=False)` on the CPU, in float64, on the **uncentered** `X`;
retain right singular vectors whose singular value
is strictly greater than `1e-10` times the largest singular value. Let `P`
project onto their orthogonal complement. Require a finite positive largest
singular value, nonzero retained rank, finite `P h`,
`||P h||_2 / ||h||_2 >= 1e-6`, and
`||X (P h)||_2 / (||X||_F ||h||_2) <= 1e-8`. A collapsed or invalid projection
is an invalid/incomplete attempt, never a reason to redraw or change the
rank cutoff. Save the singular values, rank, the retained fraction
`||P h||_2 / ||h||_2`, the discarded fraction `||h - P h||_2 / ||h||_2`,
the residual and the intended direction hashes. Their squares sum to one in
exact arithmetic.

Fix the planned update norm at `0.001 * ||W||_F`, where `W` is the original
site weight and the norm is calculated in float64. In the following fixed
order evaluate an unedited baseline and five edits at that site:

1. **Projected gradient:** scale `(P h) g^T` to the planned norm.
2. **Unprojected gradient:** scale `h g^T` to the same planned norm.
3. **Projected random:** use a dedicated `torch.Generator(device="cpu")`
   seeded `20261102` and call `torch.randn` in float64 first for a length-3072
   factor and then for a length-768 factor, with no intervening draws. Project
   the first factor with `P` and scale their outer product to that norm.
4. **Opposite:** negate the projected-gradient intended float32 update.
5. **Nuisance span:** use another dedicated CPU `torch.Generator` seeded
   `20261101` for one float64 `torch.randn((3072, rank))` draw, then compute
   `torch.linalg.qr(draw, mode="reduced")` and use its column span. Project
   `h` off that rank-matched span and scale its outer product with `g` to the
   same norm. This single draw is a descriptive control, not a random-direction
   distribution. Record the Torch version and CPU build used for both draws.

For each non-opposite edit, form and normalize the outer product in float64,
cast the intended update once to float32, then apply the float32 addition to
the original weight. The nuisance QR must be finite, with every absolute
diagonal entry of `R` strictly greater than `1e-12` times the largest absolute
diagonal entry; every projected factor must retain at least `1e-6` of its
preprojection norm. Do not
redraw failed controls. Require each actual float32 post-addition weight
difference to have a finite, nonzero float64 Frobenius norm within
`[0.999, 1.001]` of its planned norm and each control's realized norm within
the same ratio of the projected gradient's realized norm. Save both planned
and realized norms and the intended-update and realized-weight digests.

The projection removes the retained right-singular subspace. Singular vectors
discarded by the cutoff can still give a nonzero `X (P h)`, so this protocol
checks a bounded approximate null rather than assuming an exact one. If the
independently computed `G` equals `h g^T` exactly, equal-norm projected and
unprojected first-order changes have
ratio `||P h||_2 / ||h||_2`. This is an ideal rank-one approximation, not a
guaranteed ratio under the allowed `1e-5` gradient residual: at a very small
retained fraction that residual can even change the sign. The primary
predictions are the directly computed float64 inner products `<G, D>` for
each planned float64 direction, each float32 intended update promoted back to
float64, and each actual post-addition weight difference. The latter is the
linearized prediction for the update actually applied after rounding.
Report the ideal ratio, direct values, their ratio only when its denominator
is nonzero, the measured `||G-h g^T||_F`, and the additive Cauchy-Schwarz
bound `||G-h g^T||_F * ||D||_F` for the difference between `<G,D>` and
`<h g^T,D>` in each direction. Compare with actual margin changes; finite
precision, nonlinear model behavior and unseen inputs can differ.

## Before evaluation

Assemble exactly four new France-capital target paraphrases and nine new
locality cases: four other country-capital relations, four other-topic facts,
and one different relation about France. Each locality case has a frozen
single-token expected continuation; both ` London` and ` Paris` must append
as their pinned single token IDs on every target. Do not choose cases using a
model forward or any study output. Check each **new evaluation** token row
against the current known public exposure inventory and relevant custodians,
including exact and bidirectional complete-row-prefix relations, and against
the other new rows. The discovery and calibration rows are intentionally
reused public inputs; the new rows must also be disjoint from them. No unknown
history or private split is thereby certified. Publish the case manifest and
its SHA-256 before model execution; preserve failed checks.

Run discovery and calibration on the pinned original model, construct all
five edits and save a fixed pre-evaluation record binding the protocol, case
manifest, checkpoint, `X`, the full `G` digest and residual, factors,
edit/control identities, singular values, norms, all planned/intended/realized
`<G,D>` predictions and runtime. Publish that record for independent review before the
first evaluation-case forward. Preserve incomplete, failed and cancelled
attempts; do not choose a replacement dose, span, random seed or case in
response to any model output.

For each condition restore and verify the original full state, registered
buffers, flags, hooks and gradients before and after its forwards. Measure
both `X @ (float64(W_edited) - float64(W_original))` and the actual before and
after float32 `c_proj` outputs at **every calibration token**, since matmul
rounding can differ from the algebraic product. Verify the captured input
activations still match their originals and record per-token and aggregate
residuals. Also record calibration final-position logits separately; neither
the calibration null nor these outputs establishes behavior on unseen inputs.

## Evaluation and result

For all 13 new cases and all six conditions save complete final-position
float32 logits and ordered per-row hashes, then compute the shifted float64
log-softmax, target margins, expected-token NLL, KL from the unedited baseline
and smallest-ID top-token tie rule used by v2. Report every target and locality
case for every condition, not only passing aggregates. Save the float32
full gradient `G` (at most 9,437,184 bytes), `X`, `h`, `g`,
projection/control factors and the scaling information in bounded,
versioned artifacts so the intended updates can be recomputed without saving
the private original weights. Independently recomputing a realized-update dot
also requires reconstructing the original weight from the pinned checkpoint.
Also save the original and realized-weight
digests, per-condition state and buffer attestations, teardown outcome,
supervisor result, seeds, and all failure or cancellation records. An
independent reader recomputes metrics from the saved logits; a separate
reviewer challenges the math, control construction, case separation and claim.

A complete run is eligible only if the original discovery margin is negative,
at least three of four original target margins are negative, and the maximum
actual unprojected calibration-output shift exceeds `1e-6`. This maximum is
taken over all 18 tokens and all 768 float32 output coordinates after their
before/after values are promoted to float64 for subtraction. A successful
result additionally requires all of the following predeclared criteria:

- The projected edit reverses at least two initially negative target
  margins, improves the mean margin across all four targets by strictly more
  than 0.5, and exceeds the projected-random mean gain over the same four rows
  by at least 0.5. Reversal counts include only originally negative rows.
- All nine locality top tokens remain unchanged, mean locality KL is at most
  0.02, maximum locality KL at most 0.10, and mean expected-token NLL
  deterioration at most 0.05 nats.
- The maximum actual projected calibration-output shift is at most 10% of
  the unprojected shift, and projected mean evaluation-locality KL is at least
  `1e-6` lower than the unprojected mean KL.

Execution, provenance, incomplete-artifact or unverified-cleanup failures are
invalid/incomplete; otherwise a failed eligibility condition is ineligible;
otherwise any failed success criterion is negative. Only all passing gates
give success. Report the five edits and all criteria regardless of outcome;
the nuisance, random and opposite controls remain descriptive single draws.
Nine locality probes cannot establish broad capability preservation.
