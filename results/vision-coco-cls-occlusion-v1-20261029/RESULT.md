# Response-selected CLS versus spatial transfer

The prespecified primary result is **inconclusive**: all eight selected cases were limited-dose, so zero were primary-evaluable. The four cats and four dogs were selected from the published 64-image screen conditional on correct-group recognition, paired geometry and an observed animal-versus-background mask contrast B ≥ 0.05. Their later outcomes were retained; they were not replaced or counted as eight primary failures.

For each case, one absolute planned dose was shared by four same-image masked-donor transfers and three fixed random controls at block index 2 (the third encoder block). Across the 24 site-specific planned `q / receiver norm` values the range was 0.002928–0.012090, below the prespecified 0.02 floor. In all eight cases the common q was limited by the natural CLS distance to the background-masked donor. These facts identify a dose bottleneck in this sample; they do not show that either site lacks information. All seven landings per case matched q within 1% and the 20% receiver cap; the largest observed relative difference from q was below 0.00000236.

The signed drop is `log P(correct group | clean) - log P(correct group | changed)`. All individual effects from the independently recomputed 1,000-logit vectors are shown below. CA/CB are CLS transfers from animal/background-masked inputs, SA/SB the corresponding spatial transfers, and RC/RA/RB the individual CLS/animal/background random controls.

| Image | Species | Animal mask | Background mask | CA | CB | SA | SB | RC | RA | RB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 63970 | Cat | +0.008397 | -0.062451 | +0.000946 | +0.003899 | -0.000607 | +0.000207 | +0.006354 | -0.000226 | -0.000045 |
| 560923 | Cat | +0.134701 | -0.069857 | +0.002567 | -0.000008 | -0.001339 | -0.000015 | +0.001099 | -0.002891 | -0.000096 |
| 378832 | Cat | +0.362974 | -0.004697 | -0.001050 | -0.001846 | +0.002110 | +0.000011 | -0.000867 | +0.000411 | +0.000353 |
| 549410 | Cat | +0.014963 | -0.055251 | -0.001742 | -0.001245 | +0.000257 | +0.000063 | +0.000411 | +0.000172 | +0.000142 |
| 292135 | Dog | +0.504066 | -0.390149 | -0.000055 | -0.002621 | -0.003094 | -0.000589 | +0.004722 | +0.000791 | +0.000966 |
| 261664 | Dog | +1.920315 | -0.201210 | +0.006670 | +0.000878 | +0.009814 | -0.008567 | +0.009408 | -0.002198 | +0.001116 |
| 374832 | Dog | +2.504407 | +0.071262 | -0.003091 | -0.003597 | +0.002337 | +0.000758 | -0.003875 | +0.001218 | -0.000264 |
| 300322 | Dog | +0.055358 | -0.005350 | +0.000248 | -0.001016 | +0.000141 | -0.000040 | -0.000569 | -0.000074 | -0.000030 |

B is the difference between the two mask drops, T_C is CA−CB, T_S is SA−SB, and D = |B−T_S|−|B−T_C|. `A_C` is T_C minus the largest absolute random-control drop. Positive D would indicate numerical closeness of this CLS contrast to B, not mediation or recovery of the masked image.

| Image | B | T_C | T_S | A_C | D | q/n CLS | q/n animal | q/n background | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 63970 | +0.070848 | -0.002953 | -0.000814 | -0.009307 | -0.002139 | 0.012090 | 0.007889 | 0.008547 | limited-dose |
| 560923 | +0.204558 | +0.002576 | -0.001324 | -0.000315 | +0.003900 | 0.010349 | 0.006372 | 0.006786 | limited-dose |
| 378832 | +0.367671 | +0.000796 | +0.002099 | -0.000071 | -0.001303 | 0.004820 | 0.002950 | 0.002928 | limited-dose |
| 549410 | +0.070214 | -0.000496 | +0.000194 | -0.000907 | -0.000690 | 0.004939 | 0.003046 | 0.003048 | limited-dose |
| 292135 | +0.894216 | +0.002565 | -0.002505 | -0.002157 | +0.005071 | 0.006697 | 0.004156 | 0.003945 | limited-dose |
| 261664 | +2.121525 | +0.005792 | +0.018381 | -0.003615 | -0.012589 | 0.005451 | 0.003487 | 0.003374 | limited-dose |
| 374832 | +2.433145 | +0.000507 | +0.001579 | -0.003368 | -0.001073 | 0.007665 | 0.004199 | 0.004762 | limited-dose |
| 300322 | +0.060708 | +0.001264 | +0.000181 | +0.000695 | +0.001083 | 0.008239 | 0.005129 | 0.005546 | limited-dose |

The response contrasts are nonzero: for example, dog 374832 has B = +2.433145, while its CLS and spatial transfer contrasts are +0.000507 and +0.001579 at this limited dose. Across the 56 matched arms, full-vocabulary total variation from clean ranged 0.000035–0.004829, `KL(clean || changed)` ranged 0.0000000065–0.000080940, and no top-one prediction changed. Including the 16 input masks, five of 72 conditions changed top one and the maximum full-vocabulary TV was 0.665286. These are descriptive case-specific outputs, not estimates of population behavior.

[The complete screen](SCREEN.md) includes every recognition and selection exclusion. The [evaluation terminal](evaluation-terminal.json), [observer](evaluation-observer.json) and [archive index](evaluation-evidence-index.json) retain every case, control, dose, score and all 184 full-logit/activation records. Reassemble the 71 evidence parts without overwriting an existing file:

```sh
python -B restore-evaluation.py --index evaluation-evidence-index.json --output evaluation-evidence.zip
```

The restored ZIP is 60,238,362 bytes, SHA-256 `6f9c55556579475e7a6dec091e7f4277fabd47fb095d38a9791a1db2f6d8d592`, containing 402 files. The [cross-phase replay](evaluation-cross-phase-audit.json) verified that all 24 initial clean/animal/background input and logit bytes match the screen and precede patch records in the saved schedule. A separate [Decimal replay](evaluation-decimal-review.json) independently checked all 72 signed effects and eight B/T_C/T_S/D comparisons; maximum absolute discrepancy was 5.17e-16. The independent [dose, landing and decision audit](evaluation-independent-audit.json) replayed all 184 records. The forward-forbidden [preflight](evaluation-preflight-terminal.json) recorded zero submissions and the [observer](evaluation-preflight-observer.json) a clean direct exit. The actual local execution revision was `cbef7889a210e49e380be11f3b0ac948c4770bf7`, whose relevant files matched public commit `bc709949482f646ba416c119d2651a2e59b0d30a`.

The [portable scientific replay and its pinned helpers](evaluation-replay-sources.zip) require only Python's standard library. From this result directory, after reconstructing `screen-evidence.zip` and `evaluation-evidence.zip`, extract both archives and the replay ZIP to fresh directories represented by `$SCREEN`, `$EVALUATION` and `$REPLAY`, then run:

```sh
python -B "$REPLAY/vision_cls_occlusion_evaluation_independent.py" \
  --manifest ../../experiments/vision_coco_cls_occlusion_v1.json \
  --screen-data "$SCREEN" --screen-result screen-result.json --screen-archive screen-evidence.zip \
  --screen-observer screen-observer.json --screen-index screen-evidence-index.json \
  --screen-readback screen-public-readback.json --screen-revision 2b555f1ee1dacbb3dc2a880302f5fa4faea6631d \
  --evaluation-data "$EVALUATION" --evaluation-observer evaluation-observer.json \
  --evaluation-revision cbef7889a210e49e380be11f3b0ac948c4770bf7 --report "$REPLAY/new-audit.json"
python -B evaluation-decimal-replay.py "$EVALUATION" evaluation-cross-phase-audit.json "$REPLAY/new-decimal.json"
```

Outputs must not already exist. The Decimal helper pins the published cross-phase audit; the independently generated scientific report may differ in its last floating digits across math libraries. The case statuses and aggregate are the substantive checks.

This single-model sample is conditional on recognition and a positive observed mask contrast. Mid-gray masks can create artificial edges, background blocks may contain unlabeled objects, and processed-grid locations are not precise native-image causal sites. Hidden states have already mixed context; even norm-matched hybrids can be off-manifold, and 192-coordinate CLS is compared with 768-coordinate spatial blocks. One fixed random direction per site does not characterize a distribution of controls. Neither these saved-byte checks nor cooperative process receipts establish execution attestation, full process cleanup, population generalization, exclusive localization, mediation, or pretraining independence.
