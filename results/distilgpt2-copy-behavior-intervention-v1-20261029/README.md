# Baseline-screened copying interventions: primary comparison ineligible

The fixed DistilGPT-2 CPU intervention completed all 12 held-out families and all 576 planned actions. **The primary H2-versus-H10 criterion is ineligible:** only five families passed the earlier clean-behavior screen, below its required minimum of six. All 12 families passed the projected-output dose checks. Among the five supported families, three contrasts were positive and their mean was −0.000061; these numbers describe an underfilled population and are not a negative significance test. The [baseline screen](../distilgpt2-copy-behavior-screen-v1-20261029/screen/README.md) retained every supported and unsupported family before these interventions.

The protocol defines each head's D as the intervention's change in repeated-target log probability minus the mean change across the three corresponding shuffled prompts, averaged over two rotations. C is D(H10) − D(H2). The primary criterion required at least six baseline-supported, dose-valid families, a strictly positive mean C, and C > 0 in at least 75% of those families. On all 12 dose-valid families, six C values were positive and the mean was −0.006161. On the seven baseline-unsupported families, three were positive and the mean was −0.010519. Both are descriptive comparisons; H7 is one fixed same-layer active control.

| Family | Clean screen | q_out | D(H2) | D(H10) | D(H7) | C |
|---:|---|---:|---:|---:|---:|---:|
| 2 | supported | 3.466 | −0.1465 | −0.0888 | +0.0217 | +0.0578 |
| 4 | unsupported | 2.746 | −0.0594 | −0.0418 | +0.0006 | +0.0176 |
| 6 | unsupported | 5.226 | −0.0886 | −0.0802 | +0.0109 | +0.0084 |
| 8 | unsupported | 1.448 | +0.0092 | −0.0121 | +0.0291 | −0.0212 |
| 10 | supported | 1.801 | −0.0958 | −0.0794 | +0.0251 | +0.0164 |
| 12 | supported | 1.763 | −0.0258 | −0.0485 | +0.0052 | −0.0227 |
| 14 | unsupported | 2.729 | −0.1648 | −0.1934 | +0.1302 | −0.0286 |
| 16 | supported | 2.188 | −0.0078 | −0.1025 | −0.0067 | −0.0947 |
| 18 | unsupported | 9.338 | −0.5753 | −0.5931 | −0.1027 | −0.0178 |
| 20 | supported | 1.820 | −0.1051 | −0.0622 | +0.0053 | +0.0430 |
| 22 | unsupported | 4.658 | −0.1548 | −0.2126 | +0.1728 | −0.0578 |
| 24 | unsupported | 7.312 | −0.2197 | −0.1938 | +0.0276 | +0.0259 |

The [family table](families.csv) includes both rotations, full precision and limiting head/variant; the [case table](cases.csv) records all 96 prompts and 288 active arms, including target logits and probabilities, full-vocabulary rank/top ties, KL, total variation, removed input norm and output-vector cosines. Across all 12 families, mean D was −0.136195 for H2, −0.142356 for H10 and +0.026598 for H7. Across the 96 prompts per arm, H2, H10 and H7 changed the unique top token in 5, 7 and 3 cases; their mean KL(clean‖arm) was 0.006618, 0.004852 and 0.002408, and mean total variation was 0.035765, 0.031393 and 0.022523, respectively. The repeated target was the clean unique top in 15/24 held-out rotation rows.

Within each family the 24 H2/H10/H7 prompt interventions used the same projected-output norm within the fixed tolerance. Family doses ranged from 1.448 to 9.338; a first-shuffle variant set the minimum in all 12 families, with H10 limiting 11 and H2 one. The largest actual norm error divided by q_out was 8.52×10⁻⁷, the largest within-family range/q_out was 1.60×10⁻⁶, and the largest observed-versus-predicted vector residual/q_out was 4.04×10⁻⁶, all below 10⁻⁴. H2/H10 output-vector cosines ranged from 0.193 to 0.363. Equal output norms therefore leave different directions, input removals and downstream nonlinear effects; H7 is not an output-matched distribution of controls.

The two rotations predict different target words, and each target also appears in the first block of its shuffled controls. These results concern next-token scores on fixed public lists, not generated-sequence accuracy or a unique natural copying circuit. Held-out families use different token groups from development and previous interventions but share topics and templates; they were screened on clean model predictions and were selected after prior results were known. No claim is made about pretraining exposure or population-wide performance. The [protocol](../../docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md) and cases were fixed before screening; this stage's [executor and numerical evaluation](../../docs/COPY_BEHAVIOR_INTERVENTION_RUNNER_V1.md) were fixed after the clean screen and before any intervention forward. Direct summation fixes the strict C sign before displayed head effects are rounded, so displayed values can differ by an ulp from their direct contrast.

## Evidence and replay

The [stream index](stream-index.json) binds all 96 original tensor files in 33 public chunks: 154,720,512 bytes, SHA-256 `3812de817676628d22684f129d81f75384e7eff64b6f9031543d8a64f1f8e0f6`. The [compressed records](records/) include the original manifest, journal, result, checkpoint projection slices and independent replay. The [saved-byte reader](../../audits/copy_behavior_intervention/saved_independent_audit.py) independently recomputed full logits, exact landing and untouched witnesses, per-family doses, comparisons and decisions, and checked the retained projection slices against the pinned checkpoint. A separate [family-summary review](family-summary-review.json) reconstructs the arm order and contrast arithmetic from recorded comparison metadata. The [archive review](archive-review.json) checked staged chunks against the originals; the [restoration review](restoration-review.json) checked the restored files and reproduced the saved-byte replay. A separate [table review](table-review.json) checked the reported rows. The [process review](process-review.json) reconciles the closed direct-child receipt and all 100 saved-file hashes; it is a cooperative observation, not an interpreter, external-process or clock attestation. The [launch boundary](launch-boundary.json) records the v41 public-inventory and resource observation; v42's status-only update merged after launch.

From the repository root, restore the intervention into a new directory:

```text
python -B results/distilgpt2-copy-behavior-intervention-v1-20261029/restore_run.py --package results/distilgpt2-copy-behavior-intervention-v1-20261029 --output-dir <restored-intervention-dir>
```

Restore the baseline and save its independent replay as `<screen-audit.json>` using the [baseline instructions](../distilgpt2-copy-behavior-screen-v1-20261029/screen/README.md). With that restored baseline and the pinned `model.safetensors`, run:

```text
python -I -B audits/copy_behavior_intervention/saved_independent_audit.py --run <restored-intervention-dir> --cases experiments/distilgpt2_copy_behavior_screen_cases_v1.json --screen-dir <restored-screen-dir> --screen-audit <screen-audit.json> --journal-sha256 abd427d09713121c2efcb60114932bc291d96b54c8cf6f2930f6900cbac388bc --result-sha256 d598b48d32ac7fbdaf8d60c2b09f48b17471aa356408eb8a2ff5891acbcb1f88 --checkpoint <pinned-model.safetensors>
```

The [table builder](build_tables.py) uses the uncompressed result, independent replay and fixed cases. The [hash index](sha256s.json) identifies all public package files. Original closed evidence also remains on E:.
