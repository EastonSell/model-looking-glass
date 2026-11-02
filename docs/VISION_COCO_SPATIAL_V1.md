# Pretrained vision transformer: matched spatial interventions v1

This is a fixed, exploratory study of an ImageNet-pretrained vision transformer,
not a claim that an image patch or token exclusively contains an animal. The
hypothesis is that, for the selected cat/dog image pairs, replacing a
segmentation-occupied block of four early-layer token states with matched-dose
states from the opposite-species image changes the model's dog-versus-cat score
more in the donor's direction than replacing the equally sized, equal-radius
outside-animal-box block. Both directions must satisfy the rule below. A negative
result applies to these selected positions, dose and examples, not to all visual
representations or larger interventions.

## Frozen model, images and split

Use [`WinKawaks/vit-tiny-patch16-224`](https://huggingface.co/WinKawaks/vit-tiny-patch16-224)
at immutable revision `77d1485af66b34d4ed0fe95dbb0c60c7496f950b`. It is a
5.7-million-parameter, 12-layer ViT with 16x16 patches and a CLS classifier.
Its model card describes a conversion of a timm checkpoint; do not claim that
COCO images are unseen during pretraining. The exact safetensors, config and
processor SHA-256s are fixed in the case manifest. Use the Transformers 4.57.3
slow `ViTImageProcessor`, Pillow 11.3.0 and float32: decode without EXIF
transposition, convert to RGB, resize directly to 224x224 bilinearly, rescale
by 1/255, and normalize using mean and standard deviation 0.5 in all channels.
Verify decoded dimensions and each recorded JPEG and preprocessed-pixel hash.

The fixed [case manifest](../experiments/vision_coco_spatial_v1.json) is selected
without model outputs from the pinned third-party COCO 2017 validation annotation
mirror at revision `0d2b1478a7c692d6227612841b28512da3def874`, SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.
The JPEG bytes were acquired and independently checked against the HTTPS
[`merve/coco` mirror](https://huggingface.co/datasets/merve/coco/tree/9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017)
at revision `9e50abcdc1361852f34841af4939cbcd2d37c92f`; the manifest retains
the original COCO/Flickr URLs and license metadata. Mirror agreement binds the
acquired bytes but is not independent proof of official-origin authenticity.
The images are not redistributed in this repository.

Selection requires exactly one annotated cat or dog, a non-crowd polygon,
recorded public-use license, image dimensions at least 224, aspect ratio at most
2, and a box occupying 8%–70% of the original frame. COCO image 39769 is
excluded because it appears in a public example of a related ViT model card.
Each selected 32x32 input block must be wholly inside both paired animal boxes,
with at least 512 of its 1,024 pixels in each separately rasterized COCO animal
mask. Polygon coordinates are mapped independently on each image axis to
224x224 and drawn using Pillow 11.3.0's filled `ImageDraw.polygon` convention;
mask hashes, total and per-token occupancy are recorded. The equally sized
control block is wholly outside both selected animal boxes, and has the same
squared distance from image center. The deterministic hash order, greedy
pairing, tie-breaks, rejected-attempt counts, IDs, boxes, positions, licenses
and hashes are in the manifest and selector. Other annotated objects may
overlap the control block; their box IDs are recorded. A prior box-only draft
is retained as `vision_coco_box_only_draft_v0.json`: annotation-only review
showed that some nominal box-interior blocks had little animal-mask coverage,
so the mask rule was added before any pretrained forward.

The mask condition applies to each four-token block, not separately to each
token: 12 of the 24 selected image/block combinations have at least one token
with fewer than 128 annotated-animal pixels out of 256, including two tokens
with zero. The outside-selected-box block overlaps another annotated object box
in 17 of 24 images, including a bird box in one. These are annotation/box
measurements; they do not prove pixel-level object contents or absence.

There are 12 disjoint pairs: the first four are development examples and the
remaining eight are experiment-held-out examples. All are public. The complete
manifest, source and this protocol are published before the first model forward.
Development outcomes are descriptive and do not change the held-out cases,
coordinates, controls or decision rule. Any change requires a new version and
new held-out examples; no replacement follows a failed model baseline.

## Score, interventions and baseline eligibility

For all 1,000 logits `z`, compute
`m = logsumexp(z[151:269]) - logsumexp(z[281:286])`: the log ratio of the
summed ImageNet dog-breed and domestic-cat class probabilities. The exact
118 dog labels and five cat labels are recorded in the manifest. Unequal
group sizes create a prior toward the dog group and incomplete class coverage;
this ratio is not a calibrated species classifier. Retain every logit, both
group probability masses, the off-group mass and top-1 label. A primary
baseline is eligible only when the cat image has `m < 0`, the dog image has
`m > 0`, their gap `delta = m_dog - m_cat` is at least 1, and each image's
correct species group has at least 20% of the full 1,000-class probability mass.
These are exploratory gates, not evidence of correct top-1 recognition.

Capture the complete state at invocation 0 of `vit.encoder.layer.2`: three
attention blocks have already run and nine remain. The state has shape
`[1,197,192]`, with token 0 the CLS token and grid token `1 + 14*row + col`.
For each pair and each site, flatten the four 192-dimensional receiver and
same-coordinate donor vectors. Compute `d_inside`, `d_outside` in binary64
using an `fsum` of squared differences, and set `q = min(d_inside,d_outside)`.
Nonfinite activations, logits or distances make the attempt incomplete; finite
distances at most `1e-6` make the pair dose-ineligible. Record baseline
eligibility and dose matchability separately, including both reasons if both
fail. Otherwise replace each receiver block by
`float32(float64(own) + (q/d_site)*(float64(donor)-float64(own)))`. Also make
a same-inside-site random-direction control: for each of 768 coordinates use
the low bit of the first byte of SHA-256 of ASCII
`vision-coco-random-v1:{pair_id}:{side}:{index}` to select -1 or +1, multiply
by `q/sqrt(768)` in binary64, add to the receiver and cast once to float32.
Sides are `cat` and `dog`; bit zero means -1 and bit one means +1. Index is
zero-based, flattening the four tokens in row-major grid order and then the
192 hidden coordinates. Signs are fixed without inspecting model outputs.
All distance and replacement arithmetic is performed on CPU after exact
float32-to-binary64 conversion, with one float32 rounding of each result.
Check delivered norms from the saved same-forward float32 landing bits.

The separately fixed [same-species donor inventory](../experiments/vision_coco_same_species_v1.json)
chooses the lowest image ID of the same species and split whose different image
has a box containing the receiver's inside block and at least 512/1,024 mask
pixels there. It records donor identity and four occupancy counts without
consulting model outputs. At the same inside position, interpolate toward that
donor by `q / d_same`; record the donor's baseline and scaling factor. If no
geometry-qualified donor exists, its distance is finite but at most `1e-6` or
below q, or its positive delivered norm differs from q by more than
`min(max(1e-5, 1e-5*q), 0.01*q)`, report this descriptive control unavailable
for that side. Its delivered norm must also match the same-side achieved inside
norm within 1%; a mismatch makes only this optional control unavailable. Do not
extrapolate or drop the primary pair. A nonfinite state
or distance is an incomplete attempt. No donor crosses the split, and these
reused images make pair observations correlated and do not create independent
held-out examples. Before any model output, donors exist for 4/8 development
and 13/16 held-out receiver sides; further dose checks may reduce those counts.

For each side and all three deliveries, the actual concatenated float32 L2
from the same-forward pre-action value must be positive and within
`min(max(1e-5, 1e-5*q), 0.01*q)` of q. The two outside and random delivered
norms must also lie within 1% of their same-side inside norm. Otherwise report
`dose-unmatchable`; do not count it as a negative. Save original, donor,
requested and actual post-action float32 bits, aggregate and per-token L2,
coordinates and ordered capture/action/capture metadata.

Run matchable interventions for every pair, including baseline-ineligible pairs.
Those ineligible results are descriptive only: retain raw margins, group and
off-group masses and interventions, but do not divide by an unsupported gap,
call their effects donor rescue, or include them in primary counts. Run
same-forward capture repeat, null and self patches at both the inside and
outside sites, full-state self and full-state
opposite-image patch (requiring the donor full logits), the natural unscaled
inside patch, matched inside, matched outside, matched random and available
same-species matched patches. An
identity patch uses bytes saved from the initial clean capture; its
same-forward pre-action witness must equal those bytes before it counts as a
self control. An
additional donor patch of the four non-CLS tokens after the final block must
leave the CLS-only head's full logits bit-identical to the recipient baseline;
this is a structural hook/control check, not a comparison of site importance.
Require exact pre/post landing and all unselected token bits unchanged. Repeat,
null, self, final-layer and final post-clean logits must match the appropriate
baseline bit-for-bit. Failed controls or missing required bytes make the
attempt incomplete, not positive or negative. Store full logits so off-group
probability mass and conditional off-group total-variation changes can be
independently recomputed; interpret conditional distributions cautiously when
their total mass is small.

## Prespecified held-out decision

For a baseline-eligible, dose-matchable pair, write `m_cat` and `m_dog` for
the clean scores and `delta = m_dog - m_cat`. The annotation-oriented movement
is `m_patched_cat - m_cat` on the cat image and `m_dog - m_patched_dog` on
the dog image. A pair is positive only when, **in both directions**, the
inside movement is at least `0.02*delta` and exceeds the absolute outside-block
movement by at least `0.01*delta`. Otherwise that evaluable pair is negative.
The random, same-species and natural full-dose results are descriptive controls, not extra
success gates. Report all continuous effects, both directions, actual doses,
relative receiver norms, total and eligible counts, baseline and dose failures.

The primary denominator is the evaluable subset of the eight held-out pairs.
With at least four evaluable, at least 75% positive gives a positive aggregate;
fewer than 75% gives a negative aggregate. Fewer than four evaluable gives
inconclusive. Always report positive/evaluable and evaluable/8 as well as all
eight individual statuses. These numerical thresholds are prespecified
operational choices, not p-values or population-level inference.

Equal aggregate L2 does not match tokenwise allocation, perturbation direction,
or causal downstream effect. The two blocks have different learned absolute
positions even with matched radius, and the outside block can contain other
objects. Attention before the patch has already mixed context, and bilinear
input resizing blends neighboring pixels. A generic smoothing perturbation can
move both group margins toward the opposite-species label because the two
ImageNet groups have different sizes; the random-direction and off-group
diagnostics help assess disruption but do not rule it out. Even a positive
result supports only this particular contextual-token comparison.

For a concrete fabricated example satisfying the 20% gate, let clean
dog/cat/off-group masses be `(0.02, 0.25, 0.73)` for the cat and
`(0.25, 0.002, 0.748)` for the dog. Their margins are approximately -2.526
and +4.828. Mixing each 1,000-class distribution by 10% with uniform moves
the margins by approximately +0.502 and -0.194: both pass the 2%-of-gap
direction threshold despite a generic, non-donor-specific perturbation. If
outside movements were zero, they would also pass the 1% advantage rule.
This example is not a prediction about the actual model.

This v1 uses a bounded, supervised CPU owner with CUDA hidden. Model-specific
GPU supervision would delay the initial experiment; a separately supervised
GPU follow-up may be run under a new version. Preserve the actual runtime and
device. The cooperative CPU supervisor records its direct child's exit; without
an OS process job it does not attest a complete process tree. Save source,
manifest, assets, runtime, per-forward bits, failures,
controls and split status. A local source/hash check is not execution
attestation, and the held-out split is not evidence of pretraining independence.
