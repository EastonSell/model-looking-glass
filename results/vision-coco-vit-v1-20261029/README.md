# Matched spatial interventions in a pretrained vision transformer

The prespecified held-out result was **negative**: zero of four evaluable
cat–dog pairs met the two-direction inside-versus-outside rule. Four more of
the eight selected pairs failed the clean baseline gate and are not counted as
tested negatives. All eight were dose-matchable, and their interventions and
controls were retained. A failed composite rule does **not** mean the visual
interventions had no effect.

This exploratory study used the pinned ImageNet-pretrained
`WinKawaks/vit-tiny-patch16-224`, 24 public COCO-derived images, and a fixed
four-pair development/eight-pair held-out split. It patched four spatial token
states after encoder block index 2 (the third block), in both directions between
each cat–dog pair.
The score was dog-group minus cat-group log-sum-exp: 118 ImageNet dog classes
and five domestic-cat classes. The input block had at least 512 of 1,024
Pillow-rasterized pixels in the selected animal mask for each paired image;
the outside-target-box control had the same block size and center radius.
The full [protocol](../../docs/VISION_COCO_SPATIAL_V1.md),
[cases](../../experiments/vision_coco_spatial_v1.json), and
[same-species donor choices](../../experiments/vision_coco_same_species_v1.json)
were published at source commit `b56d07e16d275a0f3ee9326bcd878277b89ec99d`
before the recorded pretrained runs.

The baseline gate required a cat score below zero, a dog score above zero, a gap of at
least 1, and at least 20% probability mass in each image's expected group. For
each evaluable pair, **both** annotation-oriented inside movements had to be
at least 2% of that clean gap and at least 1% of the gap greater than the
absolute outside movement. At least four of eight pairs had to be evaluable;
75% of the evaluable pairs had to pass for a positive aggregate. These are
prespecified operational thresholds, not tests of statistical significance.

Values below are changes in log odds. A positive annotation-oriented movement
is toward the opposite-species label: patched minus clean for a cat, and clean
minus patched for a dog. Ineligible rows are descriptive; they did not enter
the primary numerator or denominator.

| Held-out pair | Status | Clean gap | Cat inside | Cat outside | Dog inside | Dog outside |
|---|---|---:|---:|---:|---:|---:|
| 04 | negative | 13.848 | +0.254 | +0.120 | +0.520 | -0.071 |
| 05 | negative | 13.589 | +0.119 | +0.050 | +0.839 | -0.154 |
| 06 | negative | 11.223 | +0.753 | +0.243 | -0.140 | +0.288 |
| 07 | negative | 10.987 | -0.154 | -0.111 | +1.025 | -0.072 |
| 08 | baseline-ineligible | 3.228 | +0.778 | +0.012 | -2.220 | -0.042 |
| 09 | baseline-ineligible | 15.710 | +0.073 | +0.007 | +0.022 | -0.057 |
| 10 | baseline-ineligible | 9.463 | +0.171 | -0.063 | -0.239 | +0.230 |
| 11 | baseline-ineligible | 7.501 | -0.052 | +0.061 | -0.875 | +0.083 |

All four excluded pairs failed the cat-group 20% mass gate; pairs 08 and 10
also failed the cat score sign. All four development pairs were likewise
baseline-ineligible, although matched descriptive interventions ran on each.
The gate did not require a correct top-1 label: pair 04's cat image passed
with 28.8% cat-group mass while its top label was `bookcase`.
Pair 04's cat inside movement, 0.254, was below the 0.277 minimum, and its
0.134 advantage over the absolute outside movement was below the 0.138 required
advantage. Pair 05's cat missed both conditions as well.
The dog movement in pair 06 and cat movement in pair 07 went in the opposite
direction from the annotation-oriented criterion.
The [full descriptive summary](descriptive-summary.json) includes every
baseline, individual condition, score, group mass, off-group distribution
change, same-species availability, and actual per-token and block dose.

The held-out common absolute dose `q` ranged from 16.297 to 18.583. This is a
large intervention relative to the receiver's four-token state: `q` divided
by the inside-state norm ranged from 0.921 to 1.249, and by the outside-state
norm from 0.990 to 1.336. The maximum relative discrepancy between requested
and realized held-out matched block doses was about `1.32e-8`. Equal aggregate
L2 does not equalize individual token changes or downstream sensitivity.

The random-direction and same-species conditions are descriptive, not extra
success gates. Only 4 of 16 held-out receiver sides had an available
same-species matched-dose intervention: three lacked a geometry-qualified
donor and nine had a natural same-species displacement smaller than `q`.
No held-out pair had this control in both directions; only two available sides
belonged to evaluable pairs (the dogs in pairs 05 and 06).
For pair 04's dog image, the random control changed the raw dog–cat margin by
-0.632, compared with -0.520 for the opposite-species inside patch. For pair
07's dog image, the inside patch raised off-group probability mass from 0.053
to 0.452. These observations are consistent with broad output disruption and
do not establish a donor-specific animal mechanism. The protocol also records
a fabricated group-size/smoothing counterexample in which nonspecific changes
can satisfy both directional criteria.

All saved repeat, null, self, full-state donor recovery, final-block and
post-clean checks passed. A separately written, model-free
[numerical replay](independent-vision-saved-audit.py) reconstructed the full
logit scores, intervention vectors and same-forward landing/untouched-token
checks, doses, pair statuses and aggregate for all 114 development and 228
held-out forwards; its [development](development-independent-audit.json) and
[held-out](heldout-independent-audit.json) reports are included. A separate
[provenance and score reader](audit_vision_coco_saved.py) checked both source
inventories, the direct-child receipts, all saved forward files and score
inputs, the model-state digest, and the held-out binding to the development
terminal. These are checks of saved records, not proof of actual execution or
independent process-tree custody. The CPU owner exited zero and was reaped in
each phase, with CUDA hidden; the process record is cooperative and does not
attest native code or all possible descendants.

The result does not localize animal information to the selected pixels. Each
block is only at least half mask-covered as a whole; individual tokens may
contain little or no annotated animal. Seventeen of 24 outside blocks overlap
another annotated object box. Layer-2 tokens have already mixed context through
attention, the inside and outside blocks use different learned absolute
positions, and interventions of this size may be off-manifold. The selected
images are public experiment-held-out cases, not known to be independent of
pretraining. Pinned third-party HTTPS image and annotation mirrors identify
and corroborate the acquired bytes; they do not independently authenticate
official COCO origin. The dog/cat groups are unequal and do not cover every breed.
The observed result concerns these positions, images, model, controls and dose,
not a population-level effect or absence of a distributed visual representation.

## Evidence and replay

The raw byte-preserving evidence is distributed as the ordered
`raw-evidence.tar.xz.part-*` files. From this directory, run
`python rebuild-evidence.py <new-archive-path>` to verify and concatenate
them, then extract with an XZ-capable tar implementation. The archive SHA-256
is `519d30de52b85194f97559a7164031b83f6465e39fb659b399965efedb3373fd`.
The [archive index](evidence-index.json) lists SHA-256 and length for all 708
original members; [part hashes](archive-parts.json) pin each published part.
The archive contains the zero-forward checkpoint preflight, source/owner
receipts, full per-forward logits and activation witnesses, pair records and
terminal records. It does not redistribute the source JPEGs or checkpoint;
the manifests identify their pinned public mirrors and hashes.

After extraction, the independent reader can be run for each split using the
checked-out source manifest and donor JSON:

```text
python independent-vision-saved-audit.py --manifest ../../experiments/vision_coco_spatial_v1.json --donors ../../experiments/vision_coco_same_species_v1.json --data <extracted>/heldout/data --split heldout --report heldout-replay.json
```

Use `--split development` and the extracted development data for the other
phase. Reports and source/manifest hashes demonstrate saved-byte consistency;
they do not independently authenticate training history, timing, or execution.
