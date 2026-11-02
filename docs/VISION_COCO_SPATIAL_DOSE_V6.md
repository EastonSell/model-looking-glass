# Screened spatial-site ViT dose response v6

The [v5 comparison](../results/vision-coco-cls-v5-20261029/README.md) found
0/2 screened pairs satisfying its matched-dose CLS-versus-spatial composite.
Its spatial doses were about 6% of the receiver-block norm. This prospective
follow-up asks whether a larger intervention at a single spatial site produces
a stronger opposite-species directional shift than its same-species and random
controls, when a smaller dose does not. It is informed by v5; thresholds are
exploratory choices, not significance tests or an independent replication.

## Fixed sample and clean screen

Reuse v5's pinned checkpoint, processor, ImageNet class groups, COCO 2017
train annotation mirror and model-blind image qualifications. Exclude all
documented v1, v3, v4 and v5 images by ID and exact JPEG and processed-pixel
hashes, including every screened image and donor. The resulting 171 prior
images are exposed development evidence. These checks do not rule out
near-duplicates, other unrecorded exposures or pretraining overlap. The mirror
and fixed hashes establish byte consistency, not authenticated COCO origin.

Select the largest non-crowd polygon instance by bounding-box area, break ties
by lowest annotation ID, then apply v5's image, box and mask qualifications
without fallback to a smaller instance; exclude mixed cat/dog images. Rank
qualifying images separately by category with SHA-256 of ASCII
`vision-coco-spatial-dose-v6:{category_id}:{image_id}`, breaking ties by
numeric image ID. Take the first 32 cats and 32 dogs; ranks 0, 4, ..., 28
(zero-indexed) are same-species donor candidates and the remaining ranks are
targets. Publish this protocol and exact selector before annotation ranking;
publish the entire fixed manifest and all JPEG/pixel hashes before screening.
Stop and retain the attempt for an insufficient inventory, failed download,
bad pixel check, or duplicate; do not substitute candidates. Never retry a
model-output attempt after an incomplete result.

Screen all 64 in fixed cat-then-dog rank order, retaining every full logit
vector and recognition failure. An image is recognized only when top-1 is in
its annotated species group and that group's mass is at least 0.20. Apply the
v5 greedy pairing, common mask-qualified 2-by-2 spatial site, recognized
same-species donors and baseline sign/gap rules, selecting at most four pairs.
Publish the entire screen, deterministic selection and saved-record audit
before interventions. Retain recognition and geometry failures, selected
targets/donors, donor reuse and all denominators. If zero pairs are selected,
report inconclusive and stop. If one to four pairs are selected, test all
selected pairs, including baseline-selected pairs that later fail a dose gate.
Before the first patch, require all selected target/donor clean logits to match
their screen bytes exactly and verify the same model state and CPU runtime.
Do not alter selection after model output. The selected sample is conditional
on recognition and geometry, not model-blind or pretraining-disjoint; the
legacy `heldout` field means this conditional evaluation phase only.

## Matched three-dose intervention

At the output of encoder block index 2, patch the selected four-token spatial
block only. This is a within-site dose response, not an alternative spatial
site or CLS-versus-spatial comparison. For each receiver use the same opposite-
species target and recognized same-species donor at every dose. Let `n` be
the receiver-block L2 norm, and `d_opp` and `d_same` the corresponding donor
distances. Compute in binary64:

```
q_high = min(0.396*n, d_opp, d_same)
q_mid  = q_high / 2
q_low  = q_high / 4
```

At each level construct an opposite-donor interpolation, same-species donor
interpolation and fixed Rademacher direction, each at that level's common
absolute q, using binary64 followed by exactly one float32 cast. Do not
extrapolate either donor. Use the *same* random signs at all three doses: the
sign at flattened coordinate `i` is positive iff the low bit of the first
SHA-256 digest byte of ASCII
`vision-coco-spatial-dose-v6:{pair_id}:{side}:{i}` is set, and negative
otherwise. One vector per side is not a random-control distribution.

A nonfinite value, or a receiver norm, donor distance or `q_low` at or below
`1e-6`, is dose-ineligible. At each level, all three float32 landings must have
positive actual L2 norm, be within
`min(max(1e-5, 1e-5*q_level), 0.01*q_level)` of the level q and within 1% of
the same-level opposite-donor actual dose; each must be no more than 40% of
its receiver norm. Check requested and landed bytes and every unselected token
in the same forward. A rounded plan failing any condition is
dose-unmatchable. Preserve q, n, both donor distances, scales, all nine actual
norms, receiver fractions, token-level norms and exclusions. Do not present a
small L2 perturbation as necessarily on-manifold.

A side is eligible for the primary comparison only if all three levels are
matchable and `q_high/n >= 0.30`. This minimum ensures the high nominal dose
is substantially above v5's approximately 0.059–0.062 spatial receiver
fraction. Among primary-eligible sides, nominal `q_low/n` is 0.075–0.099;
this low dose therefore does not repeat v5's exact spatial dose.
A finite, matchable side below this minimum is still run and reported
descriptively as `limited-dose`; it is excluded from the primary evaluable
denominator. Both sides of a pair must meet the dose rule. Record the
individual low/mid/high plans and every missing/unmatchable control. Keep
clean-repeat, null, self, full-state self and donor-recovery, final-layer
non-CLS and post-clean controls; full-state donor swaps are descriptive and
not subject to the 40% cap. Record full 1,000-class logits and same-forward
activation witnesses, full-vocabulary TV and `KL(clean || patched)`, top-1,
cat/dog/off-group mass and off-group conditional TV for every condition.
Whole-vocabulary TV includes cat/dog changes; it is not an off-target-only
measure.

## Fixed decision and limitations

Let `I_j`, `S_j`, and `R_j` be the opposite-donor, same-donor and random
changes in dog-versus-cat log-sum-exp score at level `j`, signed toward the
opposite species, and let `A_j = I_j - max(abs(S_j), abs(R_j))`. A side
supports the prespecified dose-response contrast only if all of these hold:

1. The low dose fails the earlier directional criterion
   `I_low >= 0.05 and A_low >= 0.025`.
2. The high dose has `I_high >= 0.10` and `A_high >= 0.05`.
3. `A_high - A_low >= 0.025`.

The middle dose is descriptive and no monotonicity is assumed. Both sides must
pass for a positive pair. Among pairs with both sides dose-eligible, at least
two must be evaluable and at least 75% must be positive for a positive
aggregate; otherwise the result is negative, or inconclusive with fewer than
two evaluable pairs. At two or three evaluable pairs all must pass. Retain
all selected and excluded pairs and every continuous signed `I`, `S`, `R`,
`A`, both raw `I_high - I_low` and adjusted `A_high - A_low`, and every
distribution diagnostic, including negative effects and ineligible sides.
Recompute the 0.30 eligibility gate and all side decisions independently from
the retained values; numerical comparison tolerances must not permit recorded
and recomputed values on opposite sides of any decision boundary.

This active-control comparison can distinguish the specified low/high
pattern from changes shared with these controls; it cannot prove that the
low dose was the sole cause of v5's negative result, that the effect is
animal-specific, or that perturbations remain on-manifold. A structured
opposite-species direction can induce generic smoothing or other off-group
changes not reproduced by either control. The states already mix global
context through attention, class-group sizes differ, and the high-dose
distance gate selects a subset of model-recognized cases. A positive or
negative result is conditional on this model, site, donor sample, dose and
screen, not a population or localization claim; reused donors create
dependence. The rule compares *control-adjusted* changes: `A_high` can exceed
`A_low` even when raw `I_high` is smaller than raw `I_low`. No raw monotonicity,
sample-size, significance, equivalence or absence claim is implied by the
fixed threshold.
