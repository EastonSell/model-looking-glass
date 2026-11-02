# Invariant-noun agreement intervention: public successor v2

The two preregistered descriptive criteria passed in both discovery and evaluation on the saved DistilGPT-2 outputs. Each split had 24/24 correct clean `is` versus `are` signs. In evaluation, mean sensitivity was 0.205178 for the fixed direction versus mean absolute active-control values 0.068777 and 0.071841; mean number-over-noun distance difference was 0.163363 versus 0.055743 and 0.063142. These are fixed, small, public-case comparisons, not evidence of a uniquely abstract grammatical-number representation.

The evaluation number-over-noun result was positive in only the required four of six cells: the frame 0/plural-attractor cell was **−0.008484** and the frame 1/plural-attractor cell was **exactly zero**. In those cells sensitivity was 0.005294 and 0.019948, respectively, below the corresponding v2 active-control responses 0.052645 and 0.052294. The aggregate result is not uniform across contexts. All 12 subjectless fragments and all eight standalone demonstrative controls also had positive u responses; their syntax and context differ from the target prompts and they are not a null distribution.

## What was tested

The fixed intervention adds or subtracts the previously selected direction `u` at the last-token output of layer 3 attention (`transformer.h.3.attn`), with nominal input-space L2 dose 2. The two previously fixed directions `v1` and `v2` are active equal-input-dose controls; equal L2 does not imply equal functional dose. For each prompt, the response `O` is half the difference between the `is − are` logit margins under `+2a` and `−2a`. For each pair of invariant nouns A/B, `S` averages the signed `this − these` response differences within each noun. `D_num` is the average absolute within-noun `this`/`these` difference; `D_lex` is the average absolute between-noun difference at a fixed determiner; `Delta = D_num − D_lex`. The names describe the fixed comparisons, not an isolated mechanism. A zero or negative Delta does not prove number information is absent.

Discovery uses `salmon` and `trout` in the `At sunrise` and `Before sunset` frames; evaluation uses `squid` and `shrimp` in `At midnight` and `During autumn`. Context 0 is `nearby`, 1 uses a singular attractor and 2 its plural form. Each frame has its own attractor pair, so frame and attractor-lemma effects are not separately identified, and the noun sets differ by split. `this` and `these` change the determiner and local syntax even though the noun surface is held fixed; determiner sensitivity, context effects, lexical effects and output bias remain possible. The cases are public and nonblind, and unknown pretraining exposure is not excluded. The six cells reuse forms and frames and are not six independent samples.

The earlier [v1 attempt](../distilgpt2-agreement-invariant-noun-v1/README.md) stopped incomplete after a journal append failure. Its 34 discovery inputs are treated as exposed; none was repeated in v2. The 34 retained v1 evaluation objects appeared beyond the 301 saved events and 302 owner-reported starts/returns under the sequential source contract. That supports, but does not authenticate, their nonexecution in v1. The operator reports that no incomplete-v1 logits or intervention-effect values were used to select the v2 cases.

The inherited v1 protocol called for fresh natural and injection custody observations. Under the explicit no-access instruction those requirements were **not met**. The [public-only departure](https://github.com/EastonSell/model-looking-glass/blob/ac35e97c250aeefa683d9fd4be7a3761d2fee3f9/docs/AGREEMENT_INVARIANT_NOUN_V2_PUBLIC_ONLY_DEPARTURE.md) was published before v2 forwards; it marks separately protected history and target relation unknown. The public support comparison is conditional on its recorded sampler/tokenizer preparation and is not a private membership query. Unknown or separately protected exposure history is unverified.

## Fixed decisions and all primary cells

| Split | Criterion | Mean u | Mean absolute v1 | Mean absolute v2 | Positive u cells | Decision |
|---|---|---:|---:|---:|---:|---|
| discovery | Sensitivity S | 0.263193 | 0.071132 | 0.027535 | 6/6 | pass |
| discovery | Number-over-noun Delta | 0.207970 | 0.064778 | 0.023587 | 5/6 | pass |
| evaluation | Sensitivity S | 0.205178 | 0.068777 | 0.071841 | 6/6 | pass |
| evaluation | Number-over-noun Delta | 0.163363 | 0.055743 | 0.063142 | 4/6 | pass |

Both baselines were adequate: 24/24 target signs correct in each split, no ties, and all six cells had four correct. The active-control columns above are means of the individual absolute cell values, not the absolute value of a signed mean. The table below gives signed S and Delta values for every fixed cell; [primary-cells.csv](primary-cells.csv) also retains D_num, D_lex, K_A, K_B, L and I for all three directions. Full unrounded values, all margins and additional breakdowns are in [science-replay.json](science-replay.json).

| Split | Frame | Context | u S | v1 S | v2 S | u Delta | v1 Delta | v2 Delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| discovery | 0 | none | 0.322651 | -0.070988 | -0.019462 | 0.260040 | 0.061293 | 0.009762 |
| discovery | 0 | singular | 0.144489 | -0.092762 | 0.072262 | 0.110659 | 0.092363 | 0.069757 |
| discovery | 0 | plural | 0.340815 | -0.035745 | 0.004488 | 0.293072 | 0.029850 | 0.000526 |
| discovery | 1 | none | 0.433980 | -0.094387 | -0.003674 | 0.387859 | 0.084246 | 0.000000 |
| discovery | 1 | singular | 0.073656 | -0.107066 | 0.063278 | -0.028952 | 0.098122 | 0.061476 |
| discovery | 1 | plural | 0.263570 | -0.025846 | 0.002048 | 0.225140 | 0.022793 | 0.000000 |
| evaluation | 0 | none | 0.242439 | -0.056265 | -0.042235 | 0.166885 | 0.027262 | 0.034672 |
| evaluation | 0 | singular | 0.263907 | -0.083536 | 0.123856 | 0.215979 | 0.081722 | 0.115110 |
| evaluation | 0 | plural | 0.005294 | 0.027684 | 0.052645 | -0.008484 | 0.011610 | 0.049314 |
| evaluation | 1 | none | 0.467352 | -0.104118 | -0.044343 | 0.414097 | 0.080599 | 0.039696 |
| evaluation | 1 | singular | 0.232126 | -0.134809 | 0.115673 | 0.191700 | 0.125347 | 0.098434 |
| evaluation | 1 | plural | 0.019948 | -0.006248 | 0.052294 | 0.000000 | -0.007917 | 0.041628 |

## Subjectless and demonstrative controls

These are signed `is − are` odd responses, not differences from a matched subject-deletion baseline. Discovery fragment u responses range 0.189205–0.423956 (mean 0.284868); evaluation fragment u responses range 0.160398–0.478683 (mean 0.289329). Discovery demonstratives range 0.136627–0.180433 (mean 0.157559); evaluation demonstratives range 0.120953–0.183069 (mean 0.152291). [control-responses.csv](control-responses.csv) also includes all four margins and even responses.

| Case | u odd | v1 odd | v2 odd |
|---|---:|---:|---:|
| evaluation_demonstrative_f0_these | 0.120953 | -0.020245 | 0.014492 |
| evaluation_demonstrative_f0_this | 0.140520 | -0.031265 | 0.025244 |
| evaluation_demonstrative_f1_these | 0.164623 | -0.023266 | 0.002495 |
| evaluation_demonstrative_f1_this | 0.183069 | -0.044878 | 0.017675 |
| evaluation_fragment_f0_c0 | 0.160398 | -0.002586 | 0.017408 |
| evaluation_fragment_f0_c1 | 0.287127 | -0.042393 | 0.011976 |
| evaluation_fragment_f0_c2 | 0.358822 | -0.031300 | 0.012911 |
| evaluation_fragment_f1_c0 | 0.212305 | -0.011917 | 0.007242 |
| evaluation_fragment_f1_c1 | 0.238636 | -0.055330 | -0.012495 |
| evaluation_fragment_f1_c2 | 0.478683 | -0.041973 | 0.003979 |
| v2_discovery_demonstrative_f0_these | 0.136627 | -0.029854 | 0.009159 |
| v2_discovery_demonstrative_f0_this | 0.159145 | -0.038239 | 0.021229 |
| v2_discovery_demonstrative_f1_these | 0.154030 | -0.032009 | 0.001415 |
| v2_discovery_demonstrative_f1_this | 0.180433 | -0.048492 | 0.019596 |
| v2_discovery_fragment_f0_c0 | 0.216246 | -0.020647 | 0.007227 |
| v2_discovery_fragment_f0_c1 | 0.271486 | -0.038275 | 0.005369 |
| v2_discovery_fragment_f0_c2 | 0.423956 | 0.005358 | 0.018303 |
| v2_discovery_fragment_f1_c0 | 0.255741 | -0.034084 | 0.005138 |
| v2_discovery_fragment_f1_c1 | 0.189205 | -0.027870 | 0.009731 |
| v2_discovery_fragment_f1_c2 | 0.352573 | -0.000097 | 0.008722 |

## Execution and reproducibility

The separate E:-backed CPU owner reported 612 forwards in 613 journal events, a complete candidate, and the observer and direct child each exited 0. The full source-first replay checked the saved last-position vocabulary logits, clean/no-op controls, schedule, landing witnesses, and 408/408 valid delivered vectors; realized L2 ranged from 1.9999999655 to 2.0000000315 (unrounded in the report). A separately implemented selected-logit replay agreed on all 68 cases, baseline counts, cells and decisions. The [full report](science-replay.json), [redacted replay-process observation](science-replay-observation.json), [selected-logit comparison](independent-selected-replay.json), and [independent file/receipt audit](independent-file-review.json) bind the saved data. The full replay cannot independently verify unsaved sequence-position logits; cooperative process observations do not authenticate execution or descendant drain.

Source revision `ac35e97c250aeefa683d9fd4be7a3761d2fee3f9`, fixed cases SHA-256 `188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b`, events SHA-256 `ffe761e9a02338bf6f9afbcb066197ed83a2488d02cf79986585ea8225e3fd10`, terminal SHA-256 `3184ed42ee8171ffc29c56251912df19c03b04c71b48d35522baf649dd52fdfb`, and [owner exit observation](owner-exit-observation.json) are retained. The source reconstructs v43 (`61a80aaf…`); v44 (`78d783dc…`) was separately supplied and checked as the operator's later public inventory. A [France public-input observation](france-observation.json) and [v44 addendum](france-v44-observation.json) were recorded after observer start and before the first saved tensor was observed; their chronology and no-forward statement are cooperative reports. Adapter's v44 cases were later reported forwarded after this observer ended, so their planned inventory labels describe the historical snapshot.

The observer began at 14:03:20 UTC; the pre-model admission marker was created at 14:05:43 and the model-admission marker at 14:05:57. The child ended at 14:08:14 and the observer exited at 14:08:34. These filesystem/operator timestamps indicate roughly 143 seconds of source, inventory and setup, 14 seconds to model admission, and 137 seconds through model work and child cleanup; they are not isolated benchmarks or an authenticated timing trace.

[archive-manifest.json](archive-manifest.json) binds all 1,450 ordinary files and the 11 ordered [archive parts](parts/), independently streamed against the closed originals. Joined size is 88,346,152 bytes, SHA-256 `1111e2630dc755273ff01a4a898b27b75d79171765c26489f47ee4ad513ebcfa`; manifest SHA-256 `da204b88f5c209c5c1151a51ccd600b88087cf232dbb9e3de6b32f4b0f0f4a8b`. From this directory, a fresh output can be reconstructed with:

```sh
python -I -S -B join_evidence.py --parts-dir parts --manifest archive-manifest.json --output ./attempt-v2.tar.gz
```

The joiner verifies the pinned manifest, each part and final archive. A failed write or final verification can leave a partial output; remove or rename that output before a new join. [case-margins.csv](case-margins.csv) provides all 68 cases and nine saved arms, with four margins per arm. The operator's original attempt and one-shot claim remain preserved separately.
