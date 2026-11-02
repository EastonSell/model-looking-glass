# Same-image occlusion: CLS versus spatial transfer, prospective v1

This exploratory successor is post hoc: in the completed
[occlusion study](../results/vision-coco-occlusion-v1-20261029/README.md),
animal-versus-background masking had a substantial positive contrast on two of
eight fresh images, while no spatial-transfer contrast met the fixed
threshold. For dog 470798, the mask contrast was +0.260510 and the spatial
transfer contrast was -0.081290. The present question is whether transferring
the same masked image's CLS state at encoder block index 2 (the third block)
reproduces that image-level contrast more closely than transferring the
corresponding spatial states at the same absolute dose.

This is a bounded internal intervention comparison, not proof that CLS mediates
the effect of masking or contains an exclusive object representation. The
prediction and all thresholds below are fixed before any new model forward,
including the discovery phase; the earlier study informed the question and
therefore is not an independent confirmation.

## Discovery and fresh selection

Discovery uses four already exposed images from the previous evaluation:
cats 404350 and 151497, dogs 470798 and 135116. These are the two highest
previous animal-minus-background mask contrasts within each species, so their
selection is explicitly based on previously observed model outputs. Preserve
the full discovery results, including cases below the fresh-screen threshold,
and publish them before forwarding any new evaluation image. Do not revise the
fixed rule or replace a case after inspecting discovery outcomes.

For the fresh inventory, exclude all 299 documented v1/v3/v4/v5/v6 and
occlusion-v1 images by ID, exact JPEG digest and processed-pixel digest. Reuse
the pinned checkpoint, processor, annotation source, largest bounding-box
polygon instance, mixed-species exclusion, and fixed 2-by-2 object/background
geometry of occlusion-v1. The object block is wholly inside the selected box
and contains at least 768 of its 1,024 selected-instance mask pixels. The
equal-area background block has equal squared center radius and is wholly
outside every recorded annotation box; the prior lexicographic tie rule is
unchanged. The mapping is to the resized 224-by-224 grid, not an exact
original-pixel footprint.

Before examining this new paired geometry, select the first 32 qualifying
annotation images of each species, ranked by SHA-256 of ASCII
`vision-coco-cls-occlusion-v1:{category_id}:{image_id}`, then by image ID.
Publish all 64 identities, hashes and geometry results before any fresh model
output. A selected download, hash, duplicate or incomplete preparation failure
aborts the fixed preparation without substitution. Exact disjointness does not establish absence of visual
near-duplicates, unknown local exposure or pretraining overlap; mirror hashes
establish consistency rather than authenticated upstream origin.

Run and save all 64 clean logits in fixed cat-then-dog rank order. Recognition
requires top-1 within the annotated species group and group mass at least 0.20.
For every recognized image with paired geometry, also run and retain both
same-area mid-gray masks, including all pixel bits and complete logits. For
species group `G`, define a stable `log P(G)` as in occlusion-v1 and
`drop(z) = log P(G|clean) - log P(G|z)`. Set `B` to animal-mask drop minus
background-mask drop. A fresh image is screen-eligible when recognition and
geometry pass and `B >= 0.05`. Select the first up to four screen-eligible
images per species in rank order. Record every recognition, geometry and B
failure and publish the entire screen, selected list and counts before any
fresh token transfer. The evaluation is conditional on both recognition and a
positive observed occlusion contrast; it is not representative of all images.

Recompute all selected clean and both masked pixel tensors and full logits
before the first selected-case patch. Require bitwise identity with the
published screen under the same pinned model and CPU runtime; an identity
failure makes the attempt incomplete, not a selection change. For zero selected
images stop with an inconclusive result; otherwise retain every selected case,
even if its subsequent dose is ineligible or unmatchable. No outcome-based
substitution or retry is allowed after a model output.

## Common-dose interventions and fixed prediction

Let `x_C` be the clean CLS token, `x_A` the clean four-token animal block and
`x_B` the clean four-token background block after encoder block index 2. Let
`y_CA` and `y_CB` be the CLS tokens from the animal-masked and background-
masked versions; let `y_A` and `y_B` be their corresponding masked spatial
blocks. In binary64, compute the three receiver norms `n_C,n_A,n_B`, the four
natural donor distances `d_CA,d_CB,d_A,d_B`, and

```
q = min(0.196*n_C, 0.196*n_A, 0.196*n_B, d_CA, d_CB, d_A, d_B)
```

At each site use `x + q*(y-x)/d`, calculated in binary64 and cast once to
float32; never extrapolate a donor. There are four donor arms: animal and
background donor CLS, and the corresponding animal and background spatial
donor blocks. Add one matched CLS random direction and one random direction at
each spatial site. Signs are fixed before outcomes: for image ID, site
(`cls` or `spatial`) and flattened coordinate `i`, use +1 iff the low bit of
the first SHA-256 byte of ASCII
`vision-coco-cls-occlusion-v1:{image_id}:{site}:{i}` is set. The two spatial
sites use the same 768 signs; CLS uses 192 signs. The CLS random replacement
is a single control, not two independent controls for its two donors. Its
replacement is `x + (q/sqrt(N))*sign` in binary64, cast once to float32,
with `N=192` for CLS and `N=768` for each spatial block.

All seven actual landings must be positive, within
`min(max(1e-5,1e-5*q),0.01*q)` of q, within 1% of the animal-donor CLS
landing, and at most 20% of their own clean receiver norm. Check the requested
and landed bytes and every unselected token in the same forward. Record all
four natural distances, all three norms, q, donor scales, every actual dose,
and each dose/receiver fraction. A norm, donor distance or q at or below
`1e-6` is dose-ineligible; failed rounding checks are dose-unmatchable. If q
is less than 2% of any of the three receiver norms, the case is limited-dose:
run and report it descriptively but exclude it from the primary denominator.
Any nonfinite returned tensor/logit makes the attempt incomplete and is
retained, not reclassified as a dose exclusion.

Run clean repeat, null and self patch at CLS and each spatial site, full-state
self, full-state recovery of each respective masked donor, both final-layer
non-CLS controls and post-clean repeat. Unscaled full-state swaps are
descriptive and are not bounded by the matched dose. Save full 1,000-logit
vectors; report individual drops, top-1, group and off-group masses,
whole-vocabulary TV, `KL(clean || changed)` and off-group conditional TV.

Write `T_C = drop(CLS from animal mask) - drop(CLS from background mask)` and
`T_S = drop(animal spatial transfer) - drop(background spatial transfer)`.
Let `R_C`, `R_A`, `R_B` be the three individual random drops. Define the
residual advantage `D = |B - T_S| - |B - T_C|`; positive D means the CLS
contrast is numerically closer to B. A matchable non-limited fresh case is
positive only if `T_C >= 0.025`, `T_C - max(|R_C|, |R_A|, |R_B|) >= 0.025`,
and `D >= 0.025`. These exploratory inclusive thresholds are not significance
tests. The saved-data replay must independently re-evaluate decisions from both
recomputed and reported values at these boundaries. Report B, all individual drops, T_C, T_S, both residuals and D for every
case, including all negative and limited-dose outcomes. Also report the raw
`T_C - T_S` as descriptive; it is not equivalent to residual closeness.

With at least four primary-evaluable images including at least one cat and one
dog, the aggregate is positive if at least 75% pass the joint rule and negative
otherwise. If fewer than four are evaluable or a species is missing, report
inconclusive. Preserve selected, dose-excluded, evaluable and positive counts
by species, and the entire recognition/geometry/B screen denominator.

## Scope

Mid-gray masking can introduce an artificial edge or out-of-distribution
input; background can contain unlabeled objects. The selected object square
can contain context or overlap other boxes. The model's attention has mixed
information by block 2 and hybrid token states can be off-manifold. Equal
absolute dose compares one 192-coordinate CLS token with 768 spatial
coordinates and generally different fractions of their receiver norms. The
two spatial arms also use different locations. One fixed random direction per
site is not a random-control distribution. This one-model, response-screened
study cannot establish population effects, exclusive localization, causal
mediation or pretraining independence.
