# Pretrained ViT: small-dose donor and control comparison v2

This is a preregistered, fixed-sample follow-up to the negative
[v1 study](VISION_COCO_SPATIAL_V1.md). In v1, zero of four evaluable held-out
pairs met its two-direction inside-versus-outside rule and four other pairs
failed the baseline gate. The requested dose was roughly as large as the
receiver block norm, and the random-direction and same-species results did not
establish donor specificity. All 24 v1 images, including its former held-out
images and same-species donors, are exposed development material here and may
not be reused as v2 targets or donors.

The narrow hypothesis is that an opposite-species donor at the selected
mask-occupied site moves the dog-versus-cat score toward the donor more than
three particular equal-dose controls: outside-site opposite-species transfer,
same-site same-species transfer, and a single fixed same-site random direction.
Both directions of a pair must satisfy the fixed rule below. A positive result
would compare these interventions on these selected images; it would not prove
that the selected tokens contain an animal-only feature or rule out other forms
of generic disruption. A negative result is failure of this exact composite,
not proof of an absence of distributed information or of equivalence.

## Model-blind case selection and inputs

Use the exact pretrained model, CPU runtime, class groups, annotation mirror,
image mirror, image processor, mask rasterization and asset hashes in v1. The
source checkpoint's training exposure to COCO is unknown; matching the pinned
HTTPS mirrors and hashes establishes consistency with those mirrors, not
independent authentication of official COCO origin. The published v1
manifest is an exclusion list: exclude every v1 target and donor image ID and
fail the entire preparation if any new JPEG or processed-pixel SHA-256 equals
a v1 hash or another selected image hash. This checks exact duplicates only,
not near-duplicates or pretraining independence.

Form eight fresh public evaluation pairs and two fresh same-species donors per
pair, with all 32 image IDs distinct. Use the v1 annotation-only candidate
requirements: one annotated cat or dog, non-crowd polygon, listed public-use
license, dimension/aspect/box restrictions, model-card example exclusion, and
at least 512 of 1,024 rasterized animal-mask pixels in the selected inside
32-by-32 block. For a target cat/dog pair, the inside block is wholly inside
both animal boxes, while an equal-size outside block is wholly outside both
boxes and has the same squared center radius. The outside block can overlap
other objects.

Order candidates independently in each species by SHA-256 of ASCII
`vision-coco-small-dose-v2:{category_id}:{image_id}`, with numeric image ID as
a tie-breaker. Greedily scan cat targets, then unused dog targets in that
order. For each candidate pair, scan geometrically valid inside/outside block
choices in the v1 order: increasing Manhattan separation of block origins,
then squared radius, then the ordered coordinate pair. The first choice with
an unused, distinct same-species donor for each target is accepted; each donor
must satisfy the same inside-box and 512-pixel mask rule at that target's inside
coordinates. For each species choose the first such unused candidate in its
fixed hash order. Mark both targets and both donors used and continue until
eight quadruplets are selected. If fewer than eight exist, stop with no model
forward; do not substitute selection rules. Record all IDs, hash ordering,
geometry, masks, overlaps, licenses and selection diagnostics. Images are
downloaded from the fixed mirror and decoded without EXIF transposition;
retain JPEG and processed-pixel hashes. Selection uses no model outputs.

The protocol and selection algorithm are committed before case preparation.
The complete case manifest, actual source, asset hashes and execution settings
must be publicly bound before the first pretrained forward. All eight cases
are run as the evaluation sample; v1 results are the available development
evidence. No cases, positions, donors, thresholds or controls are replaced
after any new model outcome. A failed attempt and any exposed data are retained.
Before selection-child startup, its trusted supervisor creates and checks an
exclusive empty bytecode-cache prefix, keeps it quiescent, and launches with
`-I -S -B -X pycache_prefix=<that prefix>`, placing the standard library ahead
of project/dependency paths. The child checks the project package origin before
import and compares its bound source bytes with the committed revision. These
cooperative checks do not authenticate the interpreter or dependencies, and an
in-child check alone cannot attest standard-library imports made during startup.

## Score, baseline and dose

Save all 1,000 float32 logits. The score `m` is log-sum-exp of dog logits
151:269 minus log-sum-exp of cat logits 281:286, using the stable v1 binary64
calculation. Record group and off-group probability masses and top-1 labels.
The correct-group baseline gate remains: cat `m < 0`, dog `m > 0`, clean gap
`m_dog - m_cat >= 1`, and at least 0.20 full-output probability mass on the
annotated correct group for each image. Same-species donor baselines, including
top-1 and group masses, are recorded but are not assumed to be recognized by
the model. Run finite, matchable interventions even when a target baseline is
ineligible; those outcomes are descriptive and excluded from the primary rule.

At output of `vit.encoder.layer.2`, after three attention blocks, consider the
four 192-wide token states as a single vector. For each receiver side compute
in binary64, using `sqrt(fsum((x-y)**2))` over exact float32 values:

* `r_in` and `r_out`, the receiver inside and outside block L2 norms;
* `d_in` and `d_out`, the receiver-to-opposite-species distances at the inside
  and outside coordinates; and `d_same`, the receiver-to-same-species donor
  distance at the inside coordinates.

The common requested dose for that side is exactly
`q = min(0.099*r_in, 0.099*r_out, d_in, d_out, d_same)`. A nonfinite input is
an incomplete attempt. A norm or distance at most `1e-6`, or `q <= 1e-6`, is
`dose-ineligible`; retain the reasons and other available records. The smaller
of the receiver norms thus caps q, and every donor scale is at most one. The
0.099 factor leaves rounding headroom; it does not guarantee that the resulting
state lies on the model's activation manifold.

For inside opposite-species, outside opposite-species, and inside same-species
patches, form `float32(float64(receiver) + (q/d)*(float64(donor)-float64(receiver)))`
coordinatewise, with the same operation order and one float32 rounding as v1.
For the random direction, use independent fixed signs from the low bit of the
first SHA-256 byte of ASCII
`vision-coco-small-dose-random-v2:{pair_id}:{side}:{index}`, zero meaning -1
and one +1, scaled by `q/sqrt(768)` and added at the inside site. Save planned
and same-forward actual float32 pre/post bytes, metadata, total and per-token
norms and receiver-relative doses. Require each realized dose to be positive,
within `min(max(1e-5,1e-5*q),0.01*q)` of q, within 1% of the same-side inside
realized dose, and at most 0.10 times its own receiver-site norm. If finite
rounding makes any check fail, label the side `dose-unmatchable`, retain the
failure and do not silently change q or count the side as a negative. All four
matched conditions must be available for the pair to be evaluable.
The 10% realized cap applies to these four matched conditions only. The
separately retained unscaled inside donor and full-state donor recovery are
descriptive controls and can be substantially larger interventions.

Retain the v1 repeat, both null and self patches, full-state self, full-state
opposite-donor recovery, unscaled inside donor, final-layer non-CLS and
post-clean controls. Exact landing, bitwise unchanged unselected states and
bitwise identical required control logits are mandatory. Any failed identity
or missing saved record makes the attempt incomplete. Save the full logits and
conditional off-group total-variation distance; interpret off-group conditional
changes cautiously when off-group mass is small.

## Fixed decision and reporting

Define an annotation-oriented score movement as patched minus clean on the cat
image, and clean minus patched on the dog image. For each evaluable side let
`I` be the matched inside opposite-species movement and `O`, `S`, and `R` the
outside opposite-species, inside same-species and inside fixed-random movements.
The side passes exactly when `I >= 0.05` and
`I >= max(abs(O), abs(S), abs(R)) + 0.025` in binary64. A pair passes only
when both sides pass. A baseline-ineligible, dose-ineligible or unmatchable
pair is reported separately and excluded from the primary evaluable subset.
At least four of the eight pairs must be evaluable; among them at least 75%
must pass for the aggregate to be positive. Fewer than four is inconclusive;
otherwise a lower fraction is negative. Always publish positives/evaluable,
evaluable/8, all eight statuses, both directions, every signed continuous
effect, mass/top-1/TV diagnostics, q and actual per-token doses, exclusions,
and the complete retained controls. The 0.05 and 0.025 thresholds are newly
chosen exploratory operational cutoffs informed by v1, not p-values.

The single fixed random direction is not a distribution over random patches.
The absolute-control rule is conservative: a same-species movement in the
opposite direction can defeat it too. Neither these controls nor a constant
log-group-size correction rules out structured generic smoothing or other
non-donor-specific effects. Unequal dog/cat groups, context mixed by earlier
attention, differing absolute positions, incomplete segmentation, image
resizing, selected public images and unknown pretraining exposure limit the
interpretation. Small L2 alone does not establish an on-manifold intervention.
The fixed finite sample does not support population-level inference.
