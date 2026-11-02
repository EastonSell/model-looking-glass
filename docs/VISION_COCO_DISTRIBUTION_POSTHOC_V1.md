# Post-hoc ViT output-distribution analysis

This is a descriptive, model-free follow-up to the public
[ViT COCO intervention v1](../results/vision-coco-vit-v1-20261029/README.md).
It was designed after those results were known; it is not part of that
experiment's preregistered success rule and does not constitute a new blind
test. It reads the saved logits and does not load a model or run a forward.

For each side of the eight public held-out pairs, compare the clean logits with
the same receiver's matched-dose inside, outside and random-direction
interventions. Report all sides, but keep the four pairs which passed the
original baseline gate separate from the four which did not. The two sides of a
pair are related observations, not independent model seeds. The random control
is one saved direction per side; this analysis does not average over a random
control population.

Let `p` be the softmax of the 1,000 clean logits and `q` the softmax after a
patch. For each condition we report:

* Total variation, `0.5 * sum(abs(q - p))`, across all 1,000 classes.
* The signed change of probability in the cat-or-dog group and the
  complementary off-group, and the fraction of absolute probability movement
  on off-group classes. The fraction is undefined if there is no movement.
* The original dog-versus-cat group log-odds change, oriented toward the
  opposite-species annotation, plus the top-1 class IDs and saved matched-dose
  information.
* For each receiver, cosine similarity between the inside and outside and
  between the inside and random displacement vectors `q - p`. A cosine is
  undefined if either displacement is zero.

Each side retains all four full probability vectors, per-side quantities, and
descriptive medians and counts. A high
whole-distribution cosine is not a guarantee of a shared cat/dog score effect:
the group score depends on relative dog and cat masses. In particular, changing
only logits outside both groups cannot change their log-odds. Off-group
probability movement can accompany broad changes, but it does not establish
their cause or prove a specific visual mechanism. No significance tests or
population generalization are claimed.

The original intervention's `q` was large relative to the receiver states,
the layer-2 token states were already contextualized, and the inside and
outside patches used different learned absolute positions. The 118 dog and 5
cat classes have unequal group sizes and do not cover all breeds; nonspecific
smoothing can change their aggregate score. The original same-species control
exists for only four of the sixteen sides and is not used here as a paired
control. The public experiment-held-out images are not known to be independent
of pretraining. See the original report for its complete controls and limits.

## Reproduction and limits

Rebuild the original public archive as described in the original report, to a
new location, then run from the repository root:

```text
python -m looking_glass.vision_distribution_posthoc \
  --archive <new-archive-path> \
  --index results/vision-coco-vit-v1-20261029/evidence-index.json \
  --manifest experiments/vision_coco_spatial_v1.json \
  --output <new-report-path>
```

The output path must not already exist. The reader checks pinned hashes and
lengths for the archive, manifest and index, streams the 708 indexed members
and the index itself with individual hashes, and verifies the 228 held-out forward records
against their saved binary metadata. It recomputes class scores and the
original baseline gate from 64 selected saved logit vectors and checks each
recorded matched dose against its recorded target and tolerance. The analysis
does not independently replay activations or attest model execution. The
original evidence is never edited.

The XZ decoder has a 32 MiB memory limit, decompressed output is capped at
128 MiB, and each indexed archive member is capped at 512 KiB. Only the first
4,000 logit bytes for each pending selected or unselected held-out forward are
retained; 228 pending records is the maximum. The report retains at most 64
1,000-class probability vectors. These are decoder and logical buffer limits,
not a measured process RSS ceiling. The output records the `tracemalloc` peak
for the analysis through report construction, which excludes output
serialization, native allocations, decoder state and the interpreter baseline.
Input SHA-256s and the executed analysis source SHA-256
are included in the machine-readable report.
