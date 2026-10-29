# Projected rank-one edit: public CPU study

**Outcome: ineligible.** Two of the four target prompts had a negative London-minus-Paris margin before editing; the frozen rule required at least three. The projected edit reversed zero of the initially negative target margins. Its mean four-case margin gain was +0.452523; the equal-norm unprojected edit gained +0.726508. All four prompts are included in those means, regardless of baseline sign. The outcome and raw criteria are preserved in the [saved report](evaluation/evaluation/provisional-report.json).

The protocol called for four capital paraphrases, but targets 1 and 4 are direct-capital prompts while targets 2 and 3 concern government location. This [known protocol deviation](../../docs/TARGETED_EDIT_PROJECTED_CASE_SCOPE.md) was recorded before any projected-study forward. The frozen requirement for at least two reversals among initially negative targets was not changed; its outcome on this mixed set does not establish strict conformance to the four-paraphrase instruction. Both direct and related groups are identified in the target table below and in the full CSV.

All five edits use a fixed dose of 0.1% of the Frobenius norm of the original site weight. The earlier v1 used 1%, different cases and different numerical thresholds. This is a public, non-blind next-token intervention, with unknown pretraining exposure. A London-minus-Paris margin measures a pairwise logit preference; it does not imply London is the top token, describe generated text, or establish a stored belief. The single random and nuisance controls are descriptive draws. The fixed-norm projected and unprojected edits do not have matched target efficacy, so the comparison is not a gain-matched preservation result. Nine locality prompts cannot establish broad capability preservation.

## Condition summaries

KL is KL(original || condition); NLL change is condition minus original expected-token NLL in nats, averaged over nine locality rows. Reversal counts include only originally negative target margins. Top-preserved means all nine locality top-token IDs match the original.

| Condition | Mean target gain | Reversals | Mean locality KL | Max locality KL | Mean locality NLL change | All locality tops preserved |
|---|---:|---:|---:|---:|---:|:---:|
| original | +0.000000 | 0 | 0.00000000 | 0.00000000 | +0.000000 | yes |
| projected | +0.452523 | 0 | 0.00023519 | 0.00065970 | +0.014213 | yes |
| unprojected | +0.726508 | 0 | 0.00052110 | 0.00143891 | +0.028580 | yes |
| projected_random | -0.000727 | 0 | 0.00000004 | 0.00000012 | +0.000057 | yes |
| opposite | -0.452433 | 0 | 0.00023451 | 0.00063886 | -0.013627 | yes |
| nuisance | +0.724566 | 0 | 0.00051717 | 0.00142856 | +0.028446 | yes |

## Target rows

M is London logit minus Paris logit. NLL pairs are London / Paris, in nats. Top values are full-vocabulary token IDs; baseline and projected tops are shown. The full six-condition [target metrics](target-metrics.csv) include both NLLs, KL, and top IDs for every row.

| Case | Relation | Baseline M | Projected M | Unprojected M | Baseline NLL L / P | Projected NLL L / P | Top ID baseline → projected |
|---|---|---:|---:|---:|---:|---:|---:|
| projected-target-1 | direct capital | -2.6834 | -1.7930 | -1.3846 | 7.2888 / 4.6053 | 6.8746 / 5.0816 | 262 → 262 |
| projected-target-2 | government location | +0.0735 | +0.2835 | +0.5104 | 5.7440 / 5.8175 | 5.6479 / 5.9314 | 262 → 262 |
| projected-target-3 | government location | -2.3508 | -2.0326 | -1.8617 | 3.8916 / 1.5408 | 3.7024 / 1.6699 | 6342 → 6342 |
| projected-target-4 | direct capital | +0.5332 | +0.9245 | +1.2143 | 5.3499 / 5.8830 | 5.1704 / 6.0949 | 262 → 262 |

## Locality rows

NLL values use each case’s frozen expected continuation. The full six-condition [locality metrics](locality-metrics.csv) include baseline and condition NLLs, KLs and top IDs for every row.

| Case | Expected continuation | Baseline NLL | Projected ΔNLL | Unprojected ΔNLL | Projected KL | Unprojected KL | Top ID baseline → projected |
|---|---|---:|---:|---:|---:|---:|---:|
| projected-locality-1 | Vienna | 7.0755 | +0.1333 | +0.1898 | 0.00052812 | 0.00106307 | 262 → 262 |
| projected-locality-2 | Athens | 3.7541 | +0.0509 | +0.0728 | 0.00048369 | 0.00097760 | 262 → 262 |
| projected-locality-3 | Oslo | 8.1754 | +0.0502 | +0.0742 | 0.00038279 | 0.00082868 | 262 → 262 |
| projected-locality-4 | Dublin | 5.4366 | -0.1079 | -0.1579 | 0.00065970 | 0.00143891 | 262 → 262 |
| projected-locality-5 | Celsius | 4.8137 | -0.0046 | +0.0418 | 0.00000025 | 0.00002092 | 13 → 13 |
| projected-locality-6 | yellow | 6.3494 | +0.0020 | +0.0261 | 0.00000052 | 0.00008701 | 257 → 257 |
| projected-locality-7 | months | 1.2009 | -0.0000 | +0.0091 | 0.00000000 | 0.00004991 | 1933 → 1933 |
| projected-locality-8 | full | 6.8601 | -0.0003 | -0.0055 | 0.00000016 | 0.00006546 | 262 → 262 |
| projected-locality-9 | euro | 1.4210 | +0.0042 | +0.0069 | 0.00006146 | 0.00015830 | 11063 → 11063 |

## Development and evidence

The published development construction used a fixed uncentered span of all 18 token-position activations from four previously exposed calibration prompts. The span had rank 17; the projected and unprojected maximum actual calibration-output shifts were 0.00002289 and 0.337113 respectively. The discovery margin was −1.268082 before the edit and increased by about 1.17501 under the projected edit. These are development measurements, not held-out outcomes.

The [development record](development/phase-record.json) and [evaluation record](evaluation/phase-record.json) bind the source, manifest, supervision, checkpoint and artifact digests. The evaluation directory retains all six raw logit matrices, all 78 ordered row hashes, per-condition state records and the unchanged numerical report. Independent model-free review recomputed the saved metrics and confirmed the ineligible decision; a separate review reconstructed the development projection and controls. Restoration, cleanup, runtime, source custody and chronology are cooperative recorded attestations rather than independently authenticated proof of execution or trusted timestamps.

