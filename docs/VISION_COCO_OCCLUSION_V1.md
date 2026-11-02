# Same-image occlusion and token transfer: prospective v1

The [v6 spatial-dose study](../results/vision-coco-spatial-dose-v6-20261029/README.md)
found 0/4 pairs meeting a control-adjusted directional composite, although
both opposite-donor and active-control changes could be substantial. This
separate small CPU study asks whether, in a model-recognized image, masking
an annotated animal region reduces its species-group probability more than
masking a matched background region, and whether the same ordering occurs
when corresponding internal token states are transferred from each masked
version into the clean image. The fixed prediction is exploratory and informed
by prior studies; it is not a significance test or a causal mediation test.

## Inputs and phase ordering

Reuse the pinned v6 ViT checkpoint, CPU processor and cat/dog class groups.
The ViT processor resizes directly to 224 by 224 without a crop; the model
uses a 14 by 14 grid of 16-pixel patches plus CLS token 0. An image-grid patch
at zero-indexed `(row, col)` corresponds to token `1 + 14*row + col`.
This mapping is to the resized processed grid, not an exact original-image
pixel footprint.

Discovery uses four already exposed v6 screen images, with frozen selected
instance and locations: cat 328437 at animal `(4,1)` and background `(1,4)`;
cat 142719 at `(7,12)` and `(0,7)`; dog 476736 at `(3,6)` and `(6,3)`;
dog 503893 at `(3,6)` and `(6,9)`. These were the first two recognized,
geometry-eligible images of each species in v6 rank order under the rule
below; their prior model exposure makes them development cases. The choice
used v6 annotation and clean-screen information, not their occlusion outcomes.
Retain the full discovery evidence and publish its outcome before any fresh
evaluation image is forwarded. The prediction below is fixed before discovery
and is not changed or selected based on discovery outcomes.

For evaluation, exclude all 235 documented v1/v3/v4/v5/v6 images by ID,
JPEG digest and processed-pixel digest. Use v6's annotation qualifications:
largest non-crowd polygon instance by bounding-box area, ties by lowest
annotation ID, followed by the inherited image and single-site geometry filters with no smaller-
instance fallback; exclude mixed cat/dog images. Rank candidates separately
by SHA-256 of ASCII `vision-coco-occlusion-v1:{category_id}:{image_id}`,
breaking ties by numeric image ID, and take the first 32 per species. Apply
the new paired animal/background geometry only after fixing those 64 ranks;
the inherited donor/target role labels do not affect this study. The
entire fixed 64-image inventory, hashes and geometry outcomes are public before
any evaluation forward. Retain a failed preparation without substitution.
Exact disjointness does not establish absence of near-duplicates, unknown
earlier exposure or pretraining overlap; mirror hashes establish byte
consistency rather than authenticated upstream origin.

For each image, select a 2-by-2-token animal block wholly inside the chosen
animal box, containing at least 768 of its 1,024 processed-grid mask pixels.
The background block is another 2-by-2 block wholly outside every recorded
annotation box in that image, including the selected animal box. The two
block centers must have equal squared distance from image center. Choose the
lexicographically smallest tuple `(Manhattan distance between block origins,
squared radius, animal row, animal col, background row, background col)`.
Use independently scaled bounding boxes and the union of the selected
instance's polygons rasterized with Pillow 11.3.0, as in v6. Record all other
boxes, mask occupancies, selected tokens and any object-site box overlaps.
Images without a qualifying pair remain in the inventory with geometry failure.
The background label means outside recorded boxes; unannotated objects and
context may remain. Equal area and radius do not imply matched visual content.

Screen all 64 unmasked evaluation images in the fixed cat-then-dog rank order,
saving all logits and recognition failures. Recognition requires top-1 in the
annotated group and group probability at least 0.20. Select the first up to
four recognized, geometry-qualified images of each species in their fixed rank
order; never use occlusion or patch outcomes to select or replace a case.
Publish the complete screen, selected list and counts before evaluation
occlusions or token transfers. Recompute each selected clean output before its
first intervention and require bitwise agreement with the saved screen under
the same checkpoint and CPU runtime. Run all selected cases, including those
later failing a dose rule. Report geometry-qualified, screened, recognized,
selected and evaluable counts separately by species, plus every exclusion.

## Interventions and fixed prediction

For a selected block, make an independently masked copy of the processed
float32 image by setting all three channels in that exact 32-by-32 rectangle
to positive zero. Check every outside-rectangle input bit and every inside
value; retain the clean and two masked pixel tensors and their hashes. Zero in
the normalized input corresponds to the processor's mid-gray RGB value.
Save full logits and encoder-block captures for clean, animal-mask and
background-mask forwards. Retain all cases even if a mask increases the
annotated group probability or changes top-1.

At the output of encoder block index 2 (the third block), let `x_o` and `x_b`
be the clean four-token blocks and let `y_o` and `y_b` be the corresponding
blocks from the animal-masked and background-masked versions of the same
image. In binary64, let `n_s = ||x_s||_2` and `d_s = ||y_s-x_s||_2` and set

```
q = min(0.196*n_o, 0.196*n_b, d_o, d_b)
```

At each site, transfer `x_s + q*(y_s-x_s)/d_s`, calculated in binary64 and
cast once to float32. Use the same fixed Rademacher signs at both sites for a
matched random direction: sign `i` is +1 iff the low bit of the first
SHA-256 byte of ASCII `vision-coco-occlusion-v1:{image_id}:{i}` is set,
otherwise -1. Each random replacement is `x_s + q*sign/sqrt(768)`, with one
float32 cast. Never extrapolate the occluded donor. All four actual landings
must be positive, within `min(max(1e-5,1e-5*q),0.01*q)` of q, within 1% of
the animal-transfer landing, and at most 20% of their receiver norm. Check
the requested and landed bytes and every unselected token in the same forward.
Record both natural distances, norms, q, scales, per-token actual doses and
all four realized fractions. Norm, donor distance or q at or below `1e-6` is
dose-ineligible. A rounded delivery failing a check is dose-unmatchable.
Matchable cases with q/n below 0.02 at either site are limited-dose: run and
report them descriptively, excluding them from the primary denominator.

Run clean-repeat, null and self controls at each site, a full-state self
control, full-state recovery of *each* masked donor, a final-layer non-CLS
control for each site, and post-clean repeat. Full-state donor swaps are
descriptive controls and are not subject to the matched 20% cap. A returned
nonfinite logit or activation is retained and makes the attempt incomplete;
it is not reclassified as an ordinary dose exclusion. Preserve each full
1,000-logit vector and report top-1, group masses, off-group mass,
whole-vocabulary TV and `KL(clean || changed)`, and off-group conditional TV.

For annotated species group `G`, calculate `log P(G|z)` by stable
`logsumexp(z[G]) - logsumexp(z[all 1000 classes])`, retaining the shifted
components so finite extreme logits do not underflow into an exclusion.
Define `drop(z) = log P(G|clean) - log P(G|z)`.
Let `B = drop(animal-mask) - drop(background-mask)`, `T =
drop(animal-transfer) - drop(background-transfer)`, and let `R_o` and `R_b`
be the individual matched-random drops at those sites. An evaluable image is
positive only if `B >= 0.05`, `T >= 0.025`, and
`T - max(abs(R_o), abs(R_b)) >= 0.025`. Record all individual signed drops,
`B`, `T`, random effects and their contrast; a small random contrast may hide
large individual changes. The middle and high doses of v6 are not reproduced
here. Thresholds are fixed exploratory choices, not statistical significance.

Among selected images with matchable non-limited doses, require at least four
evaluable images including at least one cat and one dog. The aggregate is
positive if at least 75% meet the joint rule, negative otherwise, and
inconclusive when the minimum is unmet. Run and report an underfilled sample
descriptively; never replace it after observing outputs. Also report the
continuous contrasts and sign agreement for all matchable images, including
negative or limited-dose cases. The joint rule is a conjunction, not a
correlation estimate or evidence of a population-wide relation.

## Interpretation

This compares one fixed fill, model, layer, annotation rule and screened
sample. Mid-gray masking can introduce an artificial edge or an out-of-
distribution image; [mask-based faithfulness evaluations can depend on such
perturbations](https://proceedings.iclr.cc/paper_files/paper/2026/hash/409fcc9d24b549969b8b9be68b56a7be-Abstract-Conference.html).
Selected object blocks can contain non-object pixels and overlap other boxes;
background blocks can contain unlabeled objects. The image-level and token-
transfer contrasts do not isolate independent visual evidence. Attention has
already mixed spatial context by block 2, and the mixed-state transplant can
be off-manifold. A positive or negative composite does not prove localization,
mediation, object-specific representation, equivalence, or absence of effect.
One random vector per image is not a random-control distribution.
