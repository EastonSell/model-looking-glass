# Public known-injection localization: two-site instrumentation check

A fixed site-blind selector identified both deliberately injected MLP sites on discovery cases and, on held-out cases, its signed effect predictions beat a zero predictor in both cells. First-changed and magnitude controls also identified both sites; the fixed constant and fixed pseudorandom choices each identified one of two. This is a small procedural instrumentation check, not a claim that a natural model circuit or numerical/linguistic capability was localized.

The study used the pinned `distilbert/distilgpt2` checkpoint at revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, on CPU with float32 and eager attention, and eight public discovery texts and eight separately fixed public held-out texts. It added +4.0 to channel 0 of the final-token MLP output at zero-based block 1 (cell A) or block 4 (cell B). Four candidate blocks were evaluated. The score is the next-token logit contrast between token IDs 318 (` is`) and 389 (` are`) on ordinary text; the texts are not claimed to form an agreement benchmark. The choice rule used only discovery score arrays, with the predeclared tie and abstention rules; public injection labels mean this is procedural, not strong human blinding. The original discovery dataset and protected one-use split were not accessed. The held-out cases are held out from this selection procedure, not from model pretraining or unknown outside exposure.

## Selection and controls

| Cell | Injected site | Selector | First changed | Largest mean activation change | Constant | Fixed random |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 1 | 1 | 1 | 1 | 2 |
| B | 4 | 4 | 4 | 4 | 1 | 4 |

The selector discovery rankings were A: 1, 4, 2, 3 and B: 4, 1, 2, 3; full scores, ties, signed predictions and baseline choices are in `discovery-freeze.json`. The fixed random choices are one realization; the analytic uniform reference is 1/4 per site, not a significance test. First-change is structurally easy here because a visible intervention is deliberately introduced into a traced path. The results comprise two fixed site trials, with eight prompt observations per phase, not sixteen independent localization trials.

Every saved null/self, clean-repeat, injection landing, unchanged-coordinate, upstream and true-site control passed the independent saved-byte audit. In particular, clean replacement at the injected site exactly restored clean logits in this deterministic setup. That restoration is an expected intervention control, not independent evidence that the selector discovered a natural causal mechanism.

## Held-out predictions and full-vocabulary distances

The table lists all four sites for each cell, including selected and unselected controls. A signed prediction is the discovery mean patch-minus-injected contrast; MAE compares that prediction to the eight held-out effects. The zero column is the corresponding MAE of predicting zero. The last column is the mean L2 distance over all 50,257 next-token logits between patched and clean output. The unpatched injected-to-clean distances were 104.30935 for A and 20.82283 for B. All individual ratios were defined; none had an injected-to-clean distance at or below the predeclared 1e-12 floor.

| Cell | Patch site | Discovery prediction | Prediction MAE | Zero MAE | Patched-to-clean L2 |
|---|---:|---:|---:|---:|---:|
| A | 1 | +0.10663 | 0.05302 | 0.06889 | 0.00000 |
| A | 2 | +0.00716 | 0.03544 | 0.03186 | 68.89533 |
| A | 3 | +0.00221 | 0.05103 | 0.05047 | 42.46317 |
| A | 4 | +0.02078 | 0.01766 | 0.01607 | 114.38047 |
| B | 1 | +0.00000 | 0.00000 | 0.00000 | 20.82283 |
| B | 2 | +0.00000 | 0.00000 | 0.00000 | 20.82283 |
| B | 3 | +0.00000 | 0.00000 | 0.00000 | 20.82283 |
| B | 4 | +0.11855 | 0.04182 | 0.09556 | 0.00000 |

At the selected site, A's prediction MAE was 0.05302 versus 0.06889 for zero, and B's was 0.04182 versus 0.09556. The fixed random control selected A site 2: its full-vocabulary distance decreased from 104.30935 to 68.89533, even though its prediction MAE (0.03544) was worse than zero (0.03186). A site 4 increased that distance to 114.38047. For B, all upstream sites 1–3 left the injected output unchanged and their zero-effect predictions tied zero, so the strict prediction-success rule does not count them as wins. These observations distinguish site identification, prediction accuracy and partial output restoration.

## Evidence and limits

Both separately observed direct children exited 0 and were reaped; each phase saved 120 actions. Discovery and held-out raw files contain 33,345,504 and 32,921,568 bytes respectively. The receipts, all 16 raw float32 files, full journals, phase summaries, frozen discovery choices and `result.json` are retained in this directory. Each action includes full-vocabulary logits and its applicable trace or intervention witnesses used in the controls. The independent replay recomputes the choices, all per-case and aggregate effects, and the controls from those bytes. Receipt and launch records are cooperative direct-process observations, not authenticated proof of every instruction or descendant process; no claim is made about unknown exposure or the protected split.

The frozen [protocol](../../docs/PUBLIC_KNOWN_INJECTION_LOCALIZATION_V1.md), [delivery clarification](../../docs/PUBLIC_KNOWN_INJECTION_DELIVERY_CLARIFICATION_V1.md), [cases](../../experiments/public_known_injection_cases_v1.json), source commit `ea3f1e78e9074e3455087919d96c84d7c20db0cc` and public pre-run precede the model work. The discovery freeze was publicly read back and independently reconstructed before the held-out start. The latest public inventory at both recorded launch checks was v35; the study's own inputs were already reserved in v34. Checkpoint staging and new evidence were directed to E: with free-space checks on E: and on the source/cache volume.
