# Same-image object occlusion and token transfer

In this fixed, model-conditioned evaluation, **0 of 8 images met the joint
prediction**. All eight were dose-matchable and evaluable, so the preregistered
aggregate is negative. Two images met the animal-versus-background occlusion
threshold, but none met the corresponding token-transfer threshold. These are
small, exploratory observations; they do not establish an absence of model
effects or of object-related information.

The [protocol](../../docs/VISION_COCO_OCCLUSION_V1.md) fixed locations,
comparisons and thresholds before any new model outcome. A 32-by-32 processed
pixel animal block and a same-area, equal-radius background block were masked
to normalized zero on each image. At encoder block index 2, the four
corresponding tokens were replaced with a dose-matched direction from the same
image's masked version; matched random directions were active controls. The
fresh [screen](screen-README.md) scored 32 cat and 32 dog images: 6 cats and
12 dogs were both recognized and geometry-qualified, and the first four of
each were selected. All eight retained their exact screen clean logits in the
evaluation. The earlier [discovery phase](discovery-README.md) used four
exposed images and was also negative; it did not change the fresh rule.

Each table entry is `log P(annotated group | clean) - log P(group | changed)`;
positive values mean a reduction in annotated-group probability. `B` is animal
mask minus background mask, and `T` is animal transfer minus background
transfer. All six signed effects are shown so that two large random effects
cannot be hidden by subtracting them.

| Image | Species | Animal mask | Background mask | Animal transfer | Background transfer | Animal random | Background random | B | T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 122196 | Cat | -0.007309 | -0.008398 | 0.003442 | -0.002054 | -0.002214 | 0.000843 | 0.001089 | 0.005496 |
| 480720 | Cat | -0.058189 | -0.045763 | 0.014041 | -0.003612 | 0.013001 | 0.005232 | -0.012426 | 0.017653 |
| 404350 | Cat | 0.001558 | -0.098187 | -0.026310 | -0.016903 | 0.002577 | -0.010714 | 0.099745 | -0.009407 |
| 151497 | Cat | -0.009216 | -0.013605 | 0.000051 | 0.000692 | 0.004080 | 0.000007 | 0.004389 | -0.000640 |
| 135116 | Dog | 0.010196 | -0.002290 | -0.000035 | -0.000114 | -0.001358 | -0.000025 | 0.012486 | 0.000079 |
| 426342 | Dog | -0.000684 | 0.000072 | 0.000128 | -0.000003 | 0.000007 | 0.000002 | -0.000756 | 0.000131 |
| 470798 | Dog | 0.224702 | -0.035809 | -0.070864 | 0.010426 | 0.010873 | 0.000415 | 0.260510 | -0.081290 |
| 468792 | Dog | -0.000949 | -0.000906 | -0.000425 | -0.000020 | 0.000124 | -0.000060 | -0.000043 | -0.000404 |

The fixed joint rule required `B >= 0.05`, `T >= 0.025`, and `T` at least
0.025 above the larger absolute random drop. `B` ranged from -0.012426 to
0.260510; `T` ranged from -0.081290 to 0.017653. For dog 470798, animal
masking reduced the correct-group probability more than background masking,
while the matched token-transfer contrast had the opposite sign. Among the
48 displayed conditions, two changed top-1 and whole-vocabulary total variation
ranged from 0.000288 to 0.454666. `KL(clean || changed)` ranged from 0.000006
to 0.506347; off-group conditional TV ranged from 0.002869 to 0.438587. These
are individual-condition ranges, not population estimates. The 32 delivered
token doses were 0.169479–0.196000 of the respective receiver-block norms,
below the 0.20 cap.

The masks can introduce artificial edges or out-of-distribution images;
background means outside annotated boxes, not necessarily devoid of objects.
The processed-grid mapping is not an exact original-pixel footprint, and
attention has mixed context by the tested block. A token transplant can be
off-manifold even at a bounded norm. This conjunction is not a correlation,
mediation or unique-localization test. One random direction per image is not a
distribution of controls. Exact disjointness from prior recorded images does
not establish near-duplicate or pretraining independence.

The [evaluation index](evaluation-evidence-index.json) lists 58 archive parts,
49,043,110 bytes in total, SHA-256
`62059785aead8bc349c32b9bea72f28249357e361bbb30fac8f600d0ec2cbf01`.
Restore the 322-file ZIP without overwriting an existing file:

```sh
python restore-evaluation.py --index evaluation-evidence-index.json --output evaluation-evidence.zip
```

The independently reviewed [scientific replay and helpers](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-occlusion-v1-20261029/discovery-independent-replay.zip)
also support evaluation data. Its [evaluation audit](evaluation-independent-audit.json)
checks every mask, active landing and control; the [cross-phase audit](screen-evaluation-independent-audit.json)
checks screen linkage and the [Decimal replay bundle](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-occlusion-v1-20261029/evaluation-decimal-replay.zip)
independently checks the signed logits and reported decisions. The
closed terminal records 144 forwards; the [direct observer](evaluation-observer.json)
records exit 0 and a reaped child with no timeout or forced stop. These are saved-byte and
cooperative-process checks, not proof of every executed instruction,
publication timing, model pretraining independence or upstream authenticity.
The two forward-forbidden preflights and the phase stdout/stderr streams are
retained in [this archive](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-occlusion-v1-20261029/phase-preflight-logs.zip). The observed local source
revisions and public commits have different commit identities; the relevant
source blobs were checked for equality at each phase.

For a saved-data replay, extract the restored evaluation ZIP, the screen ZIP
and both replay bundles into separate directories. From this result directory,
substitute those directory paths in the following commands:

```sh
python replay/vision_occlusion_intervention_independent.py --manifest ../../experiments/vision_coco_occlusion_v1.json --data evaluation-data --screen-data screen-data --report local-audit.json
python decimal/vision_occlusion_evaluation_decimal_summary.py --data evaluation-data --audit evaluation-independent-audit.json --report local-decimal.json
```

The Decimal replay pins the published audit to the original saved evidence.
The newly generated scientific report may differ slightly across math libraries
while preserving the same saved-byte checks and decisions.
