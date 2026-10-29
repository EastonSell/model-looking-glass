# Donor-context retention: fixed public cases

The fixed one-sided half-retention criterion passed in **all three evaluation pairs**. In total, 47 of 48 evaluation edges met the intervention-defined eligibility threshold, and all 47 met the half-retention inequality. The stronger full-retention sensitivity failed in all three pairs: 13 of the same 47 informative edges passed it. These are dependent edges across three public, outcome-informed pairs, not a population estimate.

## What was compared

For each recipient and an opposite-class donor cue, let L be the mask's `logit(any) - logit(some)`. W is the class-directed change in L after patching the cue state from that donor cue in the recipient's own lexical context. X is the corresponding change when the **same donor cue** comes from the alternate context, using the same recipient, cue site and requested L2 norm. The sign is preset by cue group: −1 for `never`/`rarely` recipients and +1 for `always`/`often` recipients. This is a class-directed contrast; it need not point toward the alternate donor's own clean L.

An edge is informative when W ≥ 0.02. It meets half-retention when X ≥ 0.5W; the separately fixed stronger sensitivity is X ≥ W. The comparisons are inclusive. A pair is evaluable only when its clean baseline and q=0.5 dose screens pass and at least 12 of its 16 edges are informative; it passes a given rule only if at least 12 informative edges meet it. At least two evaluable evaluation pairs and two pair passes are required for an aggregate pass. This one-sided half rule accepts X greater than W. It conditions on W, an intervention outcome, and does not subtract the active same-class or same-cue controls.

| Pair | Informative edges | Half-retained | Fully retained (X ≥ W) | Primary | Stronger sensitivity |
|---|---:|---:|---:|---|---|
| Discovery | 15/16 | 15/15 | 6/15 | pass | fail |
| Evaluation 01 | 16/16 | 16/16 | 4/16 | pass | fail |
| Evaluation 02 | 16/16 | 16/16 | 3/16 | pass | fail |
| Evaluation 03 | 15/16 | 15/15 | 6/15 | pass | fail |

All four pairs passed the baseline and dose screens at q=0.5. The excluded discovery edge had W=0.007606, X=0.003798; the excluded evaluation-03 edge had W=0.012204, X=0.012566. Both were positive but below the W threshold. Every evaluation within- and alternate-context clean donor–recipient gap happened to agree with the preset class direction; that was not guaranteed by the design. [edges.csv](edges.csv) retains all 64 W/X edges, exclusions, clean gaps, active-control effects and matched geometry; [pairs.csv](pairs.csv) retains the pair screens and counts.

## Active controls and full vocabulary

Across evaluation transfers, patched-minus-clean L ranged from −0.123355 to +0.221064 at cue2, −0.073041 to +0.054713 at object2, and −0.160754 to +0.177956 at mask2. Cue5 changes were exactly zero. Object and mask patches are active comparators, and the primary rule does not establish class specificity.

| Evaluation pair | Alternate same-cue mean absolute ΔL | Alternate same-class mean absolute ΔL | Same-cue full TV | Same-class full TV |
|---|---:|---:|---:|---:|
| 01 | 0.014784 | 0.029531 | 0.006082 | 0.006314 |
| 02 | 0.008187 | 0.015624 | 0.003611 | 0.004671 |
| 03 | 0.022799 | 0.015991 | 0.005942 | 0.006500 |

The TV values are over the full mask vocabulary and are descriptive means for matched alternate-context controls. The full vocabulary top token changed in 1 of 168 evaluation cue2 transfers, and in 0/168 at each other site. The one change was the same-cue alternate-context control for the waiter/`often` recipient: `most` to `portions`. These are transfer counts with shared recipients, not independent cases. No clean or patched top-token ties occurred. All discovery sites had 0/56 changes. [top-tokens.csv](top-tokens.csv) retains every transfer; [tv.csv](tv.csv) and [effects.csv](effects.csv) retain full and target-excluded TV, raw effects and target probability mass.

Across clean rows the `any`+`some` probability mass ranged from 0.009874 to 0.778323, and across patched rows from 0.009583 to 0.780171. Excluded-token TV renormalizes the remaining mass, so it can behave differently when that mass is small.

## Geometry and scope

The 48 evaluation donor-direction cosines ranged from 0.873941 to 0.983451 (mean 0.955071); they describe these specific matched directions and do not prove a context-invariant representation. All pairs used requested q=0.5, and saved actual transfer norms ranged from 0.499999937 to 0.500000066 with maximum relative error 1.32244e−7. The fixed public contexts change role, verb and object together; word identity, wording, donor direction and effective dose remain entangled. The design and thresholds were informed by earlier results. The cases were visible during design, are not independent population draws, and are not known to be absent from model pretraining. `rarely` differs from explicit negation, and `any` has other uses. The positive rule is not evidence of a general negation mechanism.

## Execution and evidence

The earlier attempt stopped at a prelaunch headroom guard and remains a separate [incomplete record](https://github.com/EastonSell/model-looking-glass/pull/733). This successor used the same scientific protocol and exact owner source `b6667ad5bfcadac6e0f45d4ad6530cca1f4584ca`; its separately reviewed controller and admission are [PR #739](https://github.com/EastonSell/model-looking-glass/pull/739). The operator and independent reviewers report that the source, cases and both replay methods were fixed/read back before the successor's first forward. The controller's waited exit was 0. The run saved 1,088 forwards and 1,096 events; before/after model-state digests match. The full-witness and separate score/TV replays agree on every pair and aggregate. Their saved-byte checks do not independently attest execution or pretraining history.

The full 214,978,381-byte journal has SHA-256 `cde003fb88e17fa4b9af3215e8e0ccbef23fb2d5df83e1ef859715b4bc5c17de` and is retained in the local user-facing evidence location; **there is no public journal download** in this package. The included `score-tv-subset.jsonl.gz` is an operator-derived post-outcome projection of the 1,088 selected-logit pairs and copied TV/mass metrics. An independent stream comparison matched every projected event and selected float32 pair to the closed journal and the copied metrics to the result. The subset alone cannot authenticate its derivation or reconstruct full-vocabulary TV, geometry, dose, landing or execution. [REPLAY.md](REPLAY.md) describes the public subset replay and the full-journal readers. The result, manifest, scores, geometry, per-edge and per-transfer tables, direct-exit/preflight copies and both audit reports are included; [inventory.json](inventory.json) records their sizes and hashes. Use the recorded execution commit for exact source reproduction: the source PR merge `1ad9bdfc` has different `looking_glass/intervention_plan.py` bytes from executed head `b6667ad5`.
