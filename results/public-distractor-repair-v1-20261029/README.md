# Public distractor and downstream-repair study

An independently replayed, exploratory test supports its fixed downstream-repair rule in two of three artificial injection cells. The layer-2 cell worsened on average, individual examples sometimes worsened in all three cells, and a matched rotated-direction control worsened the mean distance in every cell. These results describe this checkpoint, dose and small set of public prompts.

We used `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa` on CPU with float32, eager attention and deterministic evaluation. The [protocol](../../docs/PUBLIC_DISTRACTOR_DOWNSTREAM_REPAIR_V1.md) and [16 cases](../../experiments/public_distractor_repair_cases_v1.json) were fixed and published before these forwards. Each case ends in a separate period. We scored the preceding position `t`, and added +4 to channel 0 of an MLP output at `(L,t)` for `L=1,2,3`; a second, explicitly disclosed +4 distractor was added at block 0 on the future period position `f`.

At the scored position, distractor-only logits equaled clean logits bit-for-bit and combined logits equaled target-only logits; at the final position the distractor produced a nonzero full-vocabulary change in all eight held-out cases. All 24 held-out target-only full-vocabulary effects were nonzero. These checks are consistent with the causal mask: the future-token distractor is irrelevant to the score at `t`, not to the model generally. The 318-minus-389 quantity below is a fixed next-token logit contrast, not a claim that these ordinary sentences constitute an agreement benchmark.

For each layer the primary statistic was fixed as `R = 1 - mean(||downstream patch - clean||₂) / mean(||combined - clean||₂)` over all eight held-out cases, using the complete scored-position vocabulary. The rule required `R > 0.25` in at least two of three cells, with all denominators above `1e-12`. Both the downstream sites and the 25% cutoff were informed by earlier development results. The fixed 2-of-3 rule was supported; this is a change in average logit distance, not a percentage of a causal pathway recovered or a guarantee that every token or case improves.

| Target L | Mean combined distance | Mean downstream distance | R | Rotated-control R | Passed >0.25? |
|---:|---:|---:|---:|---:|:---:|
| 1 | 96.3779 | 54.8616 | +0.4308 | -0.5082 | yes |
| 2 | 47.6233 | 52.2733 | -0.0976 | -0.2712 | no |
| 3 | 49.7740 | 22.2287 | +0.5534 | -0.2014 | yes |

The rotated comparison is one fixed, active, cyclic channel displacement at the same downstream site; it is not a null distribution. All 24 held-out rotated displacements met the prespecified 0.1% input-norm matching criterion. Equal input L2 does not imply equal functional dose. The three aggregate rotated fractions are negative, although their individual cases vary.

Per-case fractions below are `1 - distance_after/distance_before`; negative values mean the corresponding patch increased distance. The primary rule above uses a ratio of means, not the mean of these per-case fractions.

| Held-out case | L1 downstream | L2 downstream | L3 downstream | L1 rotation | L2 rotation | L3 rotation |
|---|---:|---:|---:|---:|---:|---:|
| distractor-repair-09 | -1.1045 | -3.0230 | +0.8037 | -2.9810 | -0.0904 | +0.4482 |
| distractor-repair-10 | +0.3142 | +0.6598 | -1.3676 | -1.4099 | +0.6409 | -3.5267 |
| distractor-repair-11 | +0.3818 | -1.4670 | +0.7462 | +0.5596 | -0.4298 | +0.7242 |
| distractor-repair-12 | +0.6649 | +0.1841 | +0.1803 | -0.7102 | -0.3231 | -5.3619 |
| distractor-repair-13 | +0.5776 | -0.4100 | +0.7916 | -0.1299 | -1.4554 | +0.3309 |
| distractor-repair-14 | -0.0918 | -3.0249 | +0.7283 | -0.5645 | -2.2263 | +0.4921 |
| distractor-repair-15 | +0.4160 | +0.8847 | +0.6080 | -0.5519 | +0.2348 | -1.2481 |
| distractor-repair-16 | +0.4970 | +0.6005 | -0.1690 | -0.4018 | -0.1410 | -0.5573 |

The selector saw only the four numeric candidate-effect columns, while the Q mapping and injection roles were public. It ranked the known source first in 3/3 deliberately selected cells. First-change selected the distractor in 0/3 source hits; activation magnitude hit 1/3; constant Q3 hit 2/3 because the published mapping places the source at Q3 twice; the fixed random realization Q1/Q1/Q4 hit 1/3. A uniform 1/4 reference is descriptive for one random candidate, not a significance test. The magnitude winner for L2/L3 reflects very small realized float32 differences around the two +4 additions and should not be read as a robust magnitude advantage.

Without other metadata or priors, the two changed/not-changed activation indicators remain identical when source and distractor roles are swapped; those indicators alone do not identify which is behaviorally relevant. The position-aware causal-mask reference returns the set of the three scored-position candidates and retains the source, but does not select a unique site. Public positions, mappings, full traces and causal patch effects supply additional information, so this is not a general impossibility result for all activation data.

Each signed prediction is the discovery mean of `M_patch − M_combined`, where `M = logit(318) − logit(389)` at `t`. Held-out error uses the same eight cases for each Q. A strict reduction in mean absolute error (MAE) compared with a zero prediction is required; exact ties do not count.

| L | Q | Source? | Frozen prediction | Held-out MAE | Zero MAE | Beats zero? |
|---:|---|:---:|---:|---:|---:|:---:|
| 1 | Q1 | no | -0.035863 | 0.085307 | 0.077396 | no |
| 1 | Q2 | no | +0.000000 | 0.000000 | 0.000000 | no |
| 1 | Q3 | yes | -0.006591 | 0.094275 | 0.092305 | no |
| 1 | Q4 | no | +0.000000 | 0.000000 | 0.000000 | no |
| 2 | Q1 | yes | +0.029772 | 0.068507 | 0.072269 | yes |
| 2 | Q2 | no | +0.000000 | 0.000000 | 0.000000 | no |
| 2 | Q3 | no | +0.000000 | 0.000000 | 0.000000 | no |
| 2 | Q4 | no | -0.049269 | 0.047182 | 0.066826 | yes |
| 3 | Q1 | no | +0.000000 | 0.000000 | 0.000000 | no |
| 3 | Q2 | no | +0.003052 | 0.037589 | 0.038352 | yes |
| 3 | Q3 | yes | +0.097156 | 0.060975 | 0.120511 | yes |
| 3 | Q4 | no | +0.000000 | 0.000000 | 0.000000 | no |

The selected source predictions beat zero for L2 and L3, but not L1. Target-site clean patches exactly restored scored logits as an instrumentation control; upstream and future-distractor patches exactly matched combined scored logits. Null, self, repeat, injection delivery and untouched-coordinate checks passed. The full result JSON retains all individual candidate effects, exact zero contrasts, output distances, rotation norms/cosines and negative values.

Discovery returned 224 actions and held-out returned 224 actions. The [discovery freeze](discovery-freeze.json) and its [receipt](discovery.receipt.json) were publicly readable and independently checked before held-out started; the merged freeze commit was `a597c136dbd20facc0f19ab9b8589dfa020c4b7e`. The [held-out receipt](held-out.receipt.json), [start observation](heldout-start-observation.json), full [result](result.json), and raw journals/streams are retained here. The receipt is a cooperative direct-child observation, not independent attestation of all execution or descendants. The independent model-free auditor recomputed all 448 actions and the result from saved bytes; see [replay instructions](replay.md).

The old protected split was not accessed. The held-out prompts are held out from this particular selection procedure, not from model pretraining or unknown external exposure. The three deliberately chosen layers share eight prompts and are not independent population samples. The injection coordinates are artificial and the identity of the true source is known; this is not evidence for a natural agreement circuit.
