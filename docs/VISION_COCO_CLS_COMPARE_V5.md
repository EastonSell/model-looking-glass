# Screened CLS versus spatial ViT intervention v5

The [v4 study](../results/vision-coco-recognized-v4-20261029/README.md) found
0/4 pairs satisfying its same-site composite despite nonzero directional and
control effects. It tested a four-token spatial block only. This prospective
follow-up asks whether, on fresh screened pairs, a recognized opposite-species
donor produces a larger **control-adjusted directional advantage** at the CLS
token than at that same kind of spatial block. It does not assert that a raw
CLS effect must exceed the raw spatial effect. The 0.05 and 0.025 thresholds
are exploratory choices made after v4, not significance tests.

## Fixed inventory and clean screen

Reuse the pinned v4 checkpoint, processor, ImageNet cat/dog groups, COCO 2017
train annotation archive, annotation geometry, image filters, one-thread CPU
runtime and deterministic recognized-image selection. Exclude all v1/v3/v4
images by ID and exact JPEG and processed-pixel hashes. The 64 v4 screened
images and all earlier v1/v3 target and donor images are exposed development
evidence; no v5 evaluation image may reuse them. These checks do not rule out
near-duplicates, unrecorded earlier exposures or model-pretraining overlap.
The S3 mirror and fixed digests establish consistency with those bytes, not
independently authenticated upstream origin.

Use the v4 annotation qualifications: select the non-crowd polygon instance
with largest bounding-box area first, breaking ties by lowest annotation ID;
then apply the geometry filters to that instance without falling back to a
smaller instance. Exclude mixed cat/dog images. Rank qualifying images separately by category using the
SHA-256 of ASCII `vision-coco-cls-v5:{category_id}:{image_id}`, breaking ties
by numeric image ID. Take the first 32 cats and 32 dogs, giving zero-indexed
ranks 0, 4, ..., 28 to the same-species donor role and the other ranks to targets. Freeze
and publish this protocol and the selector before annotation ranking; publish
the full 64-image manifest and JPEG/pixel digests before any model output.
Stop and retain the attempt if a selected download or pixel check fails or
duplicates another selected or prior image, including before model screening;
do not substitute another candidate. Never retry an incomplete model-output
attempt. If the annotation inventory has fewer than 32 of
either species, stop and report the selection failure without model work.

Screen all 64 images in fixed cat-then-dog rank order with one clean forward
each, retaining every 1,000-logit vector and every recognition pass and reason
for failure. A recognized image has a top-1 class in its annotated species
group and correct-group softmax mass >=0.20. Apply v4's fixed greedy pairing,
common mask-qualified 2-by-2 spatial site, recognized same-species donors,
baseline sign and >=1 gap rules; retain up to four pairs without target reuse,
with donor reuse and rejected combinations recorded. Publish the complete
screen, deterministic selected pairs and their audit before interventions.
Screening makes the selected pairs model-output-conditioned, not a
model-blind held-out sample. They are fresh only relative to the documented
local exposures. A legacy `heldout` field denotes this conditional evaluation
phase, not pretraining independence.

If no pair is selected, record an inconclusive decision and stop. If one to
four are selected, test all selected pairs. Recompute every selected target's
and donor's clean output before the first intervention and require exact
float32-logit agreement with the saved screen and unchanged model state and
runtime. Do not change the sample after intervention output.

## Sites, dose and controls

At the output of encoder block index 2 (the third block), compare CLS token 0
against the selected four-token spatial block. For each receiver, use the same
opposite-species target and recognized same-species donor at both sites. Let
`n_C` and `n_P` be the receiver L2 norms at CLS and spatial sites, and let
`d_C_opp`, `d_C_same`, `d_P_opp`, `d_P_same` be the four receiver-to-donor
distances. Compute q and distances in binary64:

```
q = min(0.099*n_C, 0.099*n_P,
        d_C_opp, d_C_same, d_P_opp, d_P_same)
```

Each site receives a donorward opposite-species interpolation, a donorward
same-species interpolation and one fixed Rademacher direction, all at the
same `q`; construct each replacement in binary64 and cast it once to float32.
The random sign at flattened coordinate `i` is the low bit of the
first digest byte of SHA-256 of ASCII
`vision-coco-cls-v5:{pair_id}:{side}:{site}:{i}`; a set bit means +1 and an
unset bit -1. `site` is `cls` or `spatial`, in the tensor's flattened order.
The random replacement is receiver + `(q / sqrt(number of coordinates))*sign`.
Do not extrapolate either donor. A receiver norm, any of the four distances,
or q <=1e-6 is dose-ineligible. A rounded delivery is dose-unmatchable if any
of the six norms is nonpositive, differs from q by more than
`min(max(1e-5, 1e-5*q), 0.01*q)`, exceeds 10% of its own site's receiver norm,
or differs by more than 1% from the same-side CLS opposite delivery. Validate
actual landed bytes and all unselected tokens in the same forward. Preserve
q, both receiver norms, all donor distances, six actual norms and relative
fractions, every exclusion and all signed outcomes. A common absolute L2 dose
has different coordinate counts (192 versus 768) and may be a different
fraction of the receiver norms; equal L2 is not equal per-coordinate change.

Retain null and self-patch controls separately at both CLS and spatial sites,
plus the same clean-repeat, full-state self and opposite donor-recovery,
final-layer non-CLS and post-clean controls as v4. The
full-state recovery and any natural unscaled donor controls are descriptive
and may exceed the matched 10% cap. A final-layer non-CLS patch is an
implementation control for this CLS-only classifier, not evidence of spatial
localization. Capture all 1,000 logits and same-forward activation witnesses.
For each condition report raw group scores, target and off-group masses,
top-1, total variation between the complete clean/patched distributions, and
`KL(clean || patched)`; whole-vocabulary TV includes target-group changes.
Also retain off-group conditional TV. Define all softmax probabilities using
the 1,000 saved finite logits and stable binary64 arithmetic.

## Fixed decision and interpretation

Let `I`, `S`, `R` be the annotation-oriented changes in dog-versus-cat
log-sum-exp score for opposite, same-species and random interventions, where
positive is toward the opposite donor. For site `X`, define
`A_X = I_X - max(abs(S_X), abs(R_X))`. A side passes only when
`I_C >= 0.05`, `A_C >= 0.025`, and `A_C - A_P >= 0.025`. Both cat and dog
sides must pass for a positive pair. Among dose-matchable pairs, at least two
must be evaluable, and at least 75% must pass for a positive aggregate;
otherwise it is negative, or inconclusive below two evaluable pairs. At two or
three evaluable pairs, all must pass. Baseline-recognition or geometry failures
remain in the screen denominator; a contradiction to the selected baseline or
cross-phase clean identity makes the attempt incomplete, not an exclusion.
Dose-ineligible and dose-unmatchable pairs remain visible and outside the
evaluable denominator.

For every selected side, report `I_C`, `I_P`, `I_C-I_P`, all same-species and
random changes, `A_C`, `A_P`, `A_C-A_P`, all outcome and dose statuses, and
the whole-distribution diagnostics. Report 64 screen rows, recognized counts
by species and role, selected targets/donors and donor reuse, selected pairs,
evaluable pairs and positives. Reused donors and two sides are not independent
population observations. An adjusted advantage can be larger even if the raw
opposite-donor movement is smaller, so do not conflate these quantities.
One random vector per site is not a random-control distribution. CLS and
spatial states have already mixed global context through attention; unequal
class-group size, structured smoothing, other objects and off-manifold
activation shifts remain alternatives. A positive result is conditional on
this model, layer, site, dose and screened sample, and cannot prove that
information is exclusive to CLS or absent from spatial tokens. A negative
result does not prove no effect or equivalence.
