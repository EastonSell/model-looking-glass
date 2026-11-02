# Screened same-site ViT comparison: negative composite result

The preregistered matched-dose criterion was **negative**: all four selected
pairs were evaluable, and **0/4** passed both directional side rules. The
criterion required at least three of these four pairs to pass. Only the dog
side of pair 02 passed. This is a failure of the stated composite, not evidence
that the interventions had no effects or that the two species' representations
are equivalent.

The [protocol](../../docs/VISION_COCO_RECOGNIZED_SCREEN_V4.md) addresses the
[earlier v3 result](../vision-coco-same-site-v3-20261029/): it first screened
fresh annotation-selected images and required model-recognized same-species
donors. The 64 images were fixed before model screening and were disjoint from
local v1/v3 images by ID, exact JPEG and processed-pixel hashes. Screening used
the pinned `WinKawaks/vit-tiny-patch16-224` model on one CPU thread, with CUDA
hidden. The [manifest](../../experiments/vision_coco_recognized_v4.json) and
[complete screen](screen-result.json) retain every image, recognition result and
reason for failure. HTTPS mirror and digest checks establish consistency with
the pinned bytes; they do not establish official upstream authenticity, visual
near-duplicate exclusion or pretraining independence.

## Screen and dependence

Recognition required the top-1 class to belong to the annotated species group
and that group's softmax mass to be at least 0.20. Of the 64 fixed images,
37 were recognized and 27 were not:

| Role | Cats recognized | Dogs recognized |
| --- | ---: | ---: |
| Target | 9/24 | 20/24 |
| Donor | 2/8 | 6/8 |

The fixed greedy rule selected four pairs, using 8 distinct targets out of 48
target-role images and 6 distinct donors out of 16 donor-role images. Two
candidate target combinations failed the common-site requirement. Each target
appeared once. Cat donors 69231 and 341270 were each used twice; each of the
four selected dog donors was used once. The paired results and repeated donor
uses are dependent observations. The screen and resulting comparison are
model-output-conditioned. The records' `split: heldout` is a legacy schema
label and does not mean model-blind selection or a pretraining-disjoint test.

## Matched comparisons

At the selected 2-by-2 token site after encoder block index 2 (the third
block), each side received
three actual matched interventions: an opposite-species target source, a
recognized same-species donor, and one fixed random direction. The score is
`logsumexp(dog logits) - logsumexp(cat logits)`. The table shows its change
oriented toward the opposite-species source: positive means dogward for a cat
receiver or catward for a dog receiver. `I`, `S`, and `R` are the opposite,
same-species, and random changes, respectively. The preregistered side rule
was `I >= 0.05` and `I >= max(abs(S), abs(R)) + 0.025`; both sides had to
pass for a pair to pass. Values below are rounded; the complete unrounded
values, probabilities, donor predictions, per-token doses, off-group mass and
conditional total variation are in [the descriptive summary](intervention-descriptive-summary.json).

| Pair | Receiver | Target ID | Same-species donor ID | I | S | R | Side passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| 00 | cat | 405058 | 69231 | +0.002580 | +0.014406 | +0.021889 | no |
| 00 | dog | 196065 | 142769 | +0.038343 | -0.125796 | -0.019655 | no |
| 01 | cat | 135158 | 69231 | -0.078128 | -0.125470 | -0.068075 | no |
| 01 | dog | 48731 | 369521 | -0.168248 | -0.221485 | -0.074431 | no |
| 02 | cat | 228861 | 341270 | -0.003363 | +0.036027 | -0.038871 | no |
| 02 | dog | 454818 | 240902 | +0.071344 | +0.042101 | -0.042488 | yes |
| 03 | cat | 510349 | 341270 | +0.075518 | +0.239797 | +0.065740 | no |
| 03 | dog | 460995 | 284658 | -0.041101 | +0.009409 | +0.042180 | no |

All eight sides were dose-matchable. Planned dose was the minimum of 0.099
times the receiver block norm and the distances to the two donors; all eight
were bounded by the 0.099 term. Across 24 matched landings, actual dose divided
by receiver norm ranged from 0.09899999097 to 0.09900000212. The largest
relative difference from the planned dose was 9.12e-8; the largest relative
difference from the same-side opposite dose was 1.01e-7. All were below the
predeclared 10% cap and within the 1% same-side match. There were no zero-dose
or unmatched-dose exclusions. The frozen random-sign rule reused the v3 seed
prefix and pair/side IDs, so the same IDs have the same signs across studies.

The saved-record replay reconciled the 14 distinct target/donor clean outputs
with their screen vectors before intervention, exact patch landings and
untouched tokens, repeat/null/self and post-clean controls, full-state self and
donor recovery, and final-layer non-CLS controls. The unscaled natural donor
and full-state recovery controls are descriptive and are **not** subject to
the matched 10% cap; natural opposite-species donor block distances here were
0.881–1.187 times the receiver block norm. The raw intervention run retained 102 forward records
and full 1,000-logit outputs, activation witnesses and pair summaries. Across
the matched conditions, off-group mass changed by -0.026615 to +0.016799 and
off-group conditional total variation ranged 0.005782–0.148023.

These results are conditional on a small selected sample and six donors; a
single random vector is not a distribution of random controls. The unequal
cat/dog class-group sizes, structured smoothing, other objects at a site and
context mixed by preceding attention remain possible explanations of score
changes. Small L2 dose does not prove an on-manifold state. The study tests a
same-site comparison, not spatial localization or animal-specificity, and
does not justify a population claim.

## Evidence and replay

The protocol and selector were published before annotation ranking; the fixed
64-image manifest was published before screening. The screen ran from public
source `f5769fb7fa6f4672bb3eb8edeacfc55917704655`; the complete screen archive,
selected pairs and observer receipt were published at
`9d90d90176a0903c47bce584c13c839038a88bb8` before the intervention. That
commit's 23 execution-source files match the screen source. The independently
reviewed scientific replay was published in the intervening checker-only
commit after the screen run; its exact bytes were reviewed before the screen
results were interpreted. This chronology describes the retained local records
and public commit order, not a trusted timestamp attestation.

The manifest SHA-256 is
`21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb`;
the screen terminal SHA-256 is
`5aa69a7399434e775ee5b40838bba2d843c3b4d8f9b9b335fae8a2da97c28cb1`;
the intervention terminal SHA-256 is
`479f1ea67efc001457427c7337313df31fa13f095ea2250ff6bfc9d99331fd1b`.
The intervention controller and its direct child were recorded as reaped with
exit 0 and no timeout; [the observer](intervention-observer.json) and
[supervisor](intervention-supervisor.json) bind these receipts. These are
cooperative local observations, not proof of every executed instruction,
publication timing or complete process-tree/native cleanup. The two exact
supervisor receipts include numeric process IDs; no local paths are included.

The separately implemented [scientific replay](independent-scientific-replay.py)
reconciles the screen, selection, saved tensors, landings, controls and
decision. Its [intervention report](intervention-independent-audit.json) has
SHA-256 `eacf6827403103716c43ff9e78f2921e2cd3bcd405bd0a45cf4aa1ecce91061c`.
An independent [60-digit Decimal replay](intervention-decimal-replay.py)
checked all 102 score vectors and the 24 matched movements; its
[report](intervention-decimal-audit.json) has SHA-256
`1b00f2d4d93ed5208f23f833f02ae43d4901d79db0c8b7f491843088823c72a9`.
The [separate spot-check](intervention-spotcheck.json) independently checked
the fixed controls and all dose landings. Independent saved-record custody
reviews are available for the [screen](screen-independent-custody.json) and
[intervention](intervention-independent-custody.json); their `original-*-audit.py`
sources preserve the original local-layout checks and are not portable package
entry points. These audits establish consistency of retained bytes, not model
or image authenticity, execution, secrecy or chronology.

The 38 numbered intervention parts concatenate to a 31,812,134-byte ZIP with
SHA-256 `143507effc970e35d11a4aa2aa48dd1bb8b2ecbcbeba93fd6ff6e2f71d2a58f1`.
The [index](intervention-evidence-index.json) contains hashes and lengths for
every part and all 210 ZIP members; an [independent package audit](intervention-independent-package-audit.json)
also compared them with the original saved files. From the repository root, the following
stdlib-only commands restore a fresh directory and reproduce the audit:

```sh
RESULT=results/vision-coco-recognized-v4-20261029
python "$RESULT/restore-evidence.py" --output /tmp/vision-recognized-v4
python "$RESULT/independent-scientific-replay.py" \
  --manifest experiments/vision_coco_recognized_v4.json \
  --screen-data /tmp/vision-recognized-v4/screen-data \
  --public-screen-result "$RESULT/screen-result.json" \
  --public-screen-archive "$RESULT/screen-evidence.zip" \
  --public-screen-custody "$RESULT/screen-custody.json" \
  --public-screen-observer "$RESULT/screen-observer.json" \
  --screen-supervisor "$RESULT/screen-supervisor.json" \
  --intervention-data /tmp/vision-recognized-v4/intervention-data \
  --report /tmp/vision-recognized-v4-audit.json
python "$RESULT/intervention-decimal-replay.py" \
  --data /tmp/vision-recognized-v4/intervention-data \
  --audit "$RESULT/intervention-independent-audit.json" \
  --output /tmp/vision-recognized-v4-decimal.json
```

The replay requires a quiescent evidence directory and a round-to-nearest
binary64 environment. The Decimal replay pins the published scientific report;
the independently regenerated report is retained separately because small
platform math-library differences can change its bytes while its checks pass.
Neither replay loads the checkpoint or reruns images.
