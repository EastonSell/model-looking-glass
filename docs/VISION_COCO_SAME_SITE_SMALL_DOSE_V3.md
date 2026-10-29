# Small-dose ViT: same-site donor and random comparison v3

This separately preregistered version replaces the proposed v2 spatial-site
follow-up; it is a same-site comparison and cannot test inside-versus-outside
localization. The [v2 protocol](VISION_COCO_SMALL_DOSE_V2.md) and its public
preselection source at `95b9f9172db503bb30c01f44bbef46f74d0fbf6b` remain in
history. Its annotation-only run exited 1 before image downloads or any model
forward: the fixed greedy rule produced zero quadruplets, even for count one.
After the v1 exclusions, that rule had 14 mask-qualified cat and seven dog
images. Its retained selector receipt has SHA-256
`570a0fdb7f231d32bcee7d8a3a0ef5461516f89b9435eed80758b6c683f5d0bd`;
its private path-bearing stderr has SHA-256
`c1cadc4125ccf06a2e9ec001a1a46f41b1bc99cafadf2fa87cd665a208858fec`.
An annotation-only feasibility exploration found that a broader same-species
candidate rule and donor reuse can support eight same-site pairs. These changes
were made before downloading new images or seeing any v3 model output.

The hypothesis is that an opposite-species donor at the selected mask-occupied
site moves the model's dog-versus-cat score toward the donor more than two
particular equal-dose controls at that *same site*: a different same-species
donor and a single fixed random direction. Both target directions must meet the
fixed rule. This compares those interventions on these examples; it does not
prove that a token contains animal-only information or exclude other types of
generic disruption. A negative result is failure of this composite, not
equivalence or absence of a distributed visual representation.

## Fixed model-blind sample

Use the pinned v1 model, annotation and image mirrors, processor, runtime,
licenses, dimensions, aspect-ratio and selected-box area requirements,
segmentation rasterizer, class groups, and exact v1 image exclusions. Matching
third-party mirror hashes shows consistency with those mirrors, not independent
authentication of COCO origin; model pretraining exposure is unknown. Exclude
all 24 v1 target/donor IDs and reject exact JPEG or processed-pixel duplicates
of v1 or different v3 image IDs. Near-duplicate exclusion is not established.

An image may contain multiple annotated cats *or* multiple annotated dogs,
but not annotations of both species. From its non-crowd annotations with a
nonempty valid list of polygons, select the instance with largest bounding-box
area, breaking ties by lowest annotation ID; then apply all v1 image and box
restrictions to that instance. Do not replace it with a smaller instance when
it fails. Rasterize the union of every polygon of that selected instance using
the v1 convention. A qualified 2-by-2 grid block is wholly within its box and
contains at least 512 of 1,024 selected-instance mask pixels. As in the
annotation-only feasibility probe, retain only images having at least one
qualified block. There is no outside-block requirement or intervention.
Record other animal and object boxes,
including their possible overlap with the selected site.

Order candidates by the pre-existing v2 rank: SHA-256 of ASCII
`vision-coco-small-dose-v2:{category_id}:{image_id}`, with numeric ID as a
tie-break. Greedily scan unused cat targets and then unused dog targets in
that order. A target can never be an already reserved donor. For each target
pair, scan their shared qualified inside blocks by increasing squared center
radius, then coordinate. At the first block with eligible cat and dog donors,
select one donor of each matching species whose selected-instance box and mask
also qualify at that coordinate. Donors must differ from every current or
previous target. Prefer an already reserved donor, then the fixed hash rank;
new donors become reserved and cannot later be targets. Mark the targets used
and continue to eight pairs. Donors may recur across cases; all 16 target IDs
are distinct and no donor ID is a target. If the fixed rule does not yield
eight, stop without any model forward. No replacement follows a model result.

Repeated references to a donor ID must have identical JPEG and processed-pixel
hashes; distinct IDs must not share either hash. Record target/donor identities,
reuse counts, masks, all instance IDs, selected geometry, JPEG/pixel hashes,
license and exact source revision. Reuse creates dependence across pair results;
do not treat the pairs as independent population observations. The v1 images,
including its former held-out cases, remain exposed development evidence, not
v3 test images. Commit and publish this protocol and selector before preparing
cases; publish the complete fixed case manifest and source before the first
pretrained forward. Preserve every failed attempt and no-forward preflight.

## Dose, controls and decision

The v1 clean baseline gate remains unchanged: cat score `m < 0`, dog score
`m > 0`, gap at least one, and at least 20% of full-output mass on each
target's annotated species group. Save all logits, group and off-group masses,
top-1 labels, and both same-species donor baseline scores. Run finite,
matchable interventions on baseline-ineligible cases as descriptive results;
do not normalize them by an unsupported gap.

After encoder layer index 2, flatten the selected four 192-wide receiver tokens.
For each receiver side compute its block norm `r` and its distances `d_opp` and
`d_same` to the opposite-species target and same-species donor at that same
coordinate, in the exact binary64 norm order defined in v2. Use
`q = min(0.099*r, d_opp, d_same)`. Nonfinite values are incomplete; `r`, either
distance or q at most `1e-6` makes the side dose-ineligible. No donor is
extrapolated. Construct opposite and same-species replacements by the v2
binary64 interpolation and one float32 cast. Construct the one random direction
by the v2 rule and ASCII seed prefix `vision-coco-small-dose-random-v2`.
Record requested and same-forward realized activation bytes and all four
per-token norms. Each of the three realized matched doses must be positive,
within the v2 tolerance of q, within 1% of the same-side opposite dose, and
at most 0.10 of its receiver norm. Rounding failures are dose-unmatchable and
are not replaced or called negatives. The 10% cap applies only to the three
matched interventions. Unscaled opposite-donor and full-state recovery
controls may be much larger; record them as descriptive controls.

Retain repeat, null and self at the inside site, full-state self and
opposite-donor recovery, unscaled inside donor, final-layer non-CLS, and
post-clean controls. Exact selected landing, unchanged unselected token bits,
and bitwise-identical required control logits are mandatory; a failure or
missing evidence makes the attempt incomplete. Save all output logits and
off-group mass/conditional total-variation diagnostics.

For each baseline-eligible, dose-matchable side, let `I` be the
annotation-oriented opposite-donor movement, `S` the same-species movement and
`R` the random movement. A side passes when `I >= 0.05` and
`I >= max(abs(S), abs(R)) + 0.025`; both sides must pass for the pair to pass.
At least four of eight pairs must be evaluable; at least 75% of those must pass
for a positive aggregate. Otherwise the result is negative, or inconclusive
when fewer than four are evaluable. Report positives/evaluable, evaluable/8,
all eight statuses, every signed raw movement and its baseline eligibility,
all dose and control values, and all missing/ineligible reasons. The thresholds
are exploratory operational choices, not significance tests.

A single random vector is not a random-effect distribution. The absolute
same-species control can defeat the rule even when it moves in the opposite
direction. These controls do not rule out structured generic smoothing, and
normalizing unequal cat/dog group sizes only adds a constant to the score.
Earlier attention has already mixed token context, selected blocks need only
be half mask-covered in aggregate, other annotated objects can overlap, and
resizing blends pixels. Small L2 does not prove an on-manifold activation.
No population-level or pretraining-disjoint claim follows from this sample.
