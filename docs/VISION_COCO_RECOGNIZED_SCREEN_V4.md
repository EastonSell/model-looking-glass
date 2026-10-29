# Screened same-site ViT comparison v4

This prospective follow-up addresses the v3 sample's six baseline failures
and its repeatedly used annotated cat donor that the model classified as a dog.
The v1 and v3 images, outputs and intervention results are exposed development
evidence. This protocol fixes a separate screening and intervention rule before
any v4 model output. It asks whether a recognized opposite-species source at a
mask-occupied site moves the dog-versus-cat group score toward that source more
than two particular equal-dose controls at the same site: a recognized,
annotation-matched same-species donor and a fixed random direction. It is not
a spatial-localization or animal-specificity test.

## Fixed annotation-only candidate pool

Use the v1 pinned ViT checkpoint, processor, class groups and CPU runtime.
The expanded candidate pool is COCO 2017 train instance annotations, not the
previous validation pool. Fetch the public annotation ZIP from
`https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`,
which must have 252,907,541 bytes and SHA-256
`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`.
Its `annotations/instances_train2017.json` member must have SHA-256
`610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`.
Freeze and publish this protocol and the deterministic selector before the
annotation-only ranking. Fetch selected JPEGs from the corresponding pinned
HTTPS S3 train2017 paths,
record the exact JPEG and processed-pixel hashes, and bind them before screening.
These hashes establish byte consistency, not pretraining independence or an
independent authentication of every upstream annotation or image.

Exclude every v1/v3 target and donor by image ID and exact JPEG/pixel hashes.
Apply the v3 license, dimension, aspect-ratio, selected-instance, bounding-box,
mask raster and >=512/1024 selected-instance pixel requirements, with no
outside-block requirement. An image containing both cat and dog annotations is
excluded. For each eligible image record all mask-qualified 2-by-2 positions,
the selected instance and other boxes. Rank by SHA-256 of ASCII
`vision-coco-recognized-v4:{category_id}:{image_id}`, then numeric image ID.
Take the first 32 mask-qualified cats and first 32 mask-qualified dogs; stop
before screening if either group is smaller. In each category, zero-indexed
rank positions divisible by four are reserved for same-species donors, and
the remaining positions are reserved for targets. Donors can never be targets.
All 64 images and roles are fixed before loading the model. If a candidate's
image cannot be fetched/validated or shares exact JPEG/pixel bytes with an
earlier exposure or a different v4 image ID, stop and retain the failure; do
not replace it. Near-duplicates and pretraining overlap are not ruled out.

## Public clean screening and deterministic selection

Publish this protocol, selector and complete image/hash manifest before the
first v4 model forward. Screen each of the 64 fixed images exactly once in the
fixed rank order within cats followed by fixed rank order within dogs, on CPU
with CUDA hidden, retaining all 1,000 logits and
the identity, scores and reason for every pass and failure. An image is
recognized only if its clean top-1 class is in its annotated cat or dog group
and its correct-group softmax mass is >=0.20. Preserve both the individual
reasons and any failed/incomplete attempt; never retry or replace an image
after a model output. Top-1 grouping is recognition by this model, not proof
that its representation is faithful to the photo.

Use only recognized images in their fixed target or donor role. Scan cat
targets in rank order; for each cat, scan unused dog targets in rank order and
accept the first qualifying pair, reserving both targets before continuing to
the next cat. For each proposed pair require
cat score < 0, dog score > 0 and dog-minus-cat score gap >=1. Scan their common
qualified sites by increasing squared center radius, then coordinate. At the
first site with a qualifying recognized cat donor and dog donor, choose for
each species the donor at that site with smallest use count, breaking ties by
the fixed rank. A used donor may recur but is never a target. Record pair, site,
donor identities, donor use counts and the counts/reasons for rejected
combinations. Greedily retain up to four target pairs, with no target reuse.
If no pair is found, declare the planned decision inconclusive and stop before
interventions. If one to four are found, use all selected pairs. A single pair
is descriptive and cannot yield a conclusive aggregate. No change or replacement
may follow an intervention result.

The complete screen records, screening source and deterministic selected-case
manifest must be committed and publicly observable before the first v4
intervention forward. In that later phase, repeat the clean forward for every
selected target and donor and require exact float32-logit equality to its saved
screening vector; bind and record the model state and runtime for both phases.
These deliberate clean control repeats do not authorize screening retries. The
screen is model-output based, so results are
conditional on that selection and are not a model-blind held-out test. The v4
images are disjoint from local v1/v3 discovery by ID and exact bytes, but may
have appeared in model pretraining.

## Matched interventions and decision

Use v3's encoder-layer-2 same-site dose and landing rules unchanged: on each
receiver side, q = min(0.099 times the receiver block norm, distance to the
opposite target, distance to the same-species donor), computed in binary64;
construct donor interpolations and the fixed Rademacher random vector with one
float32 cast. As in v3, a receiver norm, opposite- or same-species donor
distance, or q at or below 1e-6 is dose-ineligible, including tiny nonzero
values. Each of the three realized matched doses must be positive, within
the v3 tolerance and 1% of the same-side opposite dose, and no more than 10%
of receiver norm. No donor is extrapolated. Rounding failures are retained as
dose-unmatchable, not counted as negative; at least two evaluable pairs are
required. Retain exact landing and unchanged-token witnesses, repeat/null/self,
full-state self and donor recovery, natural unscaled inside donor, final-layer
non-CLS and post-clean controls and all logits. The 10% cap applies only to
the three matched interventions, not the unscaled recovery controls.
The fixed random direction reuses v3's `vision-coco-small-dose-random-v2:`
prefix, pair ID and side; therefore an equal pair ID and side have the same
sign vector across v3 and v4. Report zero donor distances and unmatched doses
for every selected side rather than suppressing them.

For each evaluable side let I, S and R be the annotation-oriented changes for
the opposite, same-species and random interventions. The side passes if
I >= 0.05 and I >= max(abs(S), abs(R)) + 0.025; both sides must pass for a
pair. At least two evaluable pairs are required; >=75% must pass for a positive
aggregate. Otherwise report negative, or inconclusive below two evaluable.
At two or three evaluable pairs the 75% rule requires every pair to pass.
These exploratory thresholds are not significance tests. Retain all selected
pairs and dose exclusions, all signed scores, class probabilities, top-1 IDs,
off-group mass and conditional total variation, donor predictions, reuse and
all screened exclusions. Report 64 screened images, including 48 target-role
and 16 donor-role images; separately report selected pairs, distinct selected
targets out of 48 and distinct selected donors out of 16, donor reuse counts,
evaluable/selected pairs and positives/evaluable pairs. Do not treat reused donors or the two sides as
independent population observations.

One fixed random direction is not a random-control distribution. Unequal
cat/dog class-group sizes and structured smoothing can change the group score,
and earlier attention has mixed token context; other objects can overlap the
site. A small activation norm does not establish an on-manifold state. Failure
of this composite is not equivalence or evidence of absence of a distributed
visual representation. No population or pretraining-disjoint claim follows.
