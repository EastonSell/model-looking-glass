# DistilGPT2 copying: pairwise and dose comparisons

On eight fixed public prompt families, the H2+H10 intervention reduced the repeated-versus-shuffled target log-probability contrast on average at both doses. The high-dose mean change was **−0.138452 nats**, with seven family values negative and one positive. The corresponding interaction was **−0.013103 nats**; the same calculation for a hypothetical additive-logit reference was **+0.008047**, leaving a pair-minus-reference contrast of **−0.021151**. These are descriptive output-scale comparisons, not proof that the heads directly cooperate. The other fixed head pairs showed different signs and family-level variation.

All 15 intervention arms left the target's top-one status unchanged on every prompt: clean and each arm had **5/8** repeated targets and **1/24** shuffled targets top-ranked. Target rank changed in 206 of the 480 arm/prompt comparisons. These fixed public examples are exploratory and nonblind; they are not a random sample, an independent model replication, or evidence of a complete circuit.

## Design and outcome

The [protocol](https://github.com/EastonSell/model-looking-glass/blob/8be210ce88a536dc9efcd769e40a8175b9beddfa/docs/DISTILGPT2_COPY_HEAD_PAIRS_DOSE_V1.md) was published before the [32 cases](https://github.com/EastonSell/model-looking-glass/blob/0b045df6bb67c7a7f900d1b51e16cf956b99c9df/experiments/distilgpt2_copy_head_pairs_dose_v1.json) were assembled. A [numerical clarification](https://github.com/EastonSell/model-looking-glass/blob/f5fe8441ff34d4eb2913ac2e369db92c3dbe21f1/docs/DISTILGPT2_COPY_PAIRS_DOSE_NUMERIC_CLARIFICATION_V1.md) was published after case assembly and before this run. It preserves the literal Python `**2` reduction while correcting an overbroad rounding claim. The 32 rows were reserved in reviewed [exposure inventory v15](https://github.com/EastonSell/model-looking-glass/blob/75491173df44b1af7a662ec56b77636a26459dd3/looking_glass/pretrained_exposures_v15.py). This chronology is recorded by the operator; the receipt is not a trusted timestamp attestation.

For each four-prompt family, the repeated prompt and three fixed shuffled controls use the same tokens and target. Let `Δ` be intervention minus clean target log probability, and define `D = Δ(repeated) − mean(Δ(shuffles))`. Thus a negative `D` does not imply that the repeated prompt's own log probability necessarily fell. A clean repeated-minus-shuffled contrast averaged **+4.028660 nats** across the families.

The following table gives the eight-family means. `Repeat Δ` and `shuffle Δ` are shown separately; the last column repeats the contrast using the unnormalized target logit. All values are intervention minus clean.

| Arm | Repeat Δ log p | Shuffle Δ log p | D log p | D target logit |
|---|---:|---:|---:|---:|
| H2 low | −0.042572 | −0.006528 | −0.036044 | −0.041891 |
| H2 matched | −0.063717 | −0.011536 | −0.052181 | −0.079355 |
| H2 high | −0.097524 | −0.021242 | −0.076282 | −0.149011 |
| H7 low | −0.001655 | +0.002797 | −0.004452 | +0.025363 |
| H7 matched | −0.003880 | +0.004095 | −0.007974 | +0.034194 |
| H7 high | −0.008465 | +0.006351 | −0.014816 | +0.045530 |
| H10 low | −0.022896 | −0.001904 | −0.020992 | −0.031943 |
| H10 matched | −0.034912 | −0.003259 | −0.031654 | −0.085085 |
| H10 high | −0.054854 | −0.005787 | −0.049067 | −0.188559 |
| H2+H7 low | −0.045118 | −0.005188 | −0.039930 | −0.013080 |
| H2+H7 high | −0.113013 | −0.022987 | −0.090026 | −0.066594 |
| H2+H10 low | −0.071287 | −0.011922 | −0.059365 | −0.161119 |
| H2+H10 high | −0.183296 | −0.044844 | −0.138452 | −0.525212 |
| H7+H10 low | −0.023619 | −0.002487 | −0.021132 | −0.005352 |
| H7+H10 high | −0.060802 | −0.014669 | −0.046133 | −0.097030 |

## Pair interactions and controls

For a same-dose pair, `J = (D_pair − D_single_a) − D_single_b`. The additive reference constructs each hypothetical full-vocabulary vector as `(logits_a − logits_clean) + logits_b` in the specified binary64 order, then applies log-softmax without rounding that vector back to float32. `J_add` is the same interaction calculated from this hypothetical vector. It is not the exact normalization component of observed `J`; the observed pair-minus-reference difference can still include downstream nonlinearities and rounding. H7 pairs are fixed comparators, not a calibrated randomization distribution.

| Pair | Dose | D pair | J | J_add | D pair − D add | Target-logit J |
|---|---|---:|---:|---:|---:|---:|
| H2+H7 | low | −0.039930 | +0.000566 | −0.000097 | +0.000664 | +0.003447 |
| H2+H7 | high | −0.090026 | +0.001071 | +0.000317 | +0.000754 | +0.036886 |
| H2+H10 | low | −0.059365 | −0.002329 | +0.002129 | −0.004458 | −0.087285 |
| H2+H10 | high | −0.138452 | −0.013103 | +0.008047 | −0.021151 | −0.187642 |
| H7+H10 | low | −0.021132 | +0.004312 | +0.000642 | +0.003669 | +0.001228 |
| H7+H10 | high | −0.046133 | +0.017749 | +0.002965 | +0.014785 | +0.045999 |

At low dose, the requested total concatenated input L2 of a pair matches each constituent head's separate `matched` arm. The comparison changes with the chosen head:

| Low-dose pair | D pair | D matched first head | D matched second head |
|---|---:|---:|---:|
| H2+H7 | −0.039930 | −0.052181 | −0.007974 |
| H2+H10 | −0.059365 | −0.052181 | −0.031654 |
| H7+H10 | −0.021132 | −0.007974 | −0.031654 |

Equal requested input L2 does not match perturbation direction, head identity, removed fraction, projection-output magnitude or downstream effect. A pair at per-head dose `q` has requested combined input L2 `√2 q`; high-dose pairs therefore have no equal-total-dose single in this design. The family minimum `m` ranged from 0.383274 to 3.854762. Low, matched-single and high per-head doses were respectively `m/2`, `√2(m/2)` and `m`. Across prompts and heads, requested removed fractions ranged from 0.025345 to 0.5 at low dose and 0.050690 to 1.0 at high dose. A family minimum couples the dose to all four prompts and all three heads, so changing a shuffle could change the repeated prompt's intervention. The maximum relative single-head achieved-dose error was 6.25×10⁻⁷; the largest absolute difference between a low pair's achieved total L2 and its matched single was 1.16×10⁻⁷. Every delivered single-head slice changed all 64 float32 elements. Full requested and achieved values, fractions, changed-element and actual bit counts are retained below.

## Family variation

These are the individual family `D` values in log-probability units. The individual repeated and three shuffled components, target-logit effects, and all prompt scores and top-one indicators are in the tables linked below.

| Family | H2 low | H2 match | H2 high | H7 low | H7 match | H7 high | H10 low | H10 match | H10 high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.033163 | −0.047411 | −0.068009 | −0.015860 | −0.022820 | −0.032863 | −0.007871 | −0.012347 | −0.020229 |
| 2 | −0.013192 | −0.018828 | −0.026979 | +0.022341 | +0.030976 | +0.042525 | −0.030984 | −0.044122 | −0.063002 |
| 3 | −0.063180 | −0.090442 | −0.129904 | −0.078524 | −0.112380 | −0.161396 | −0.037635 | −0.055523 | −0.083369 |
| 4 | −0.061919 | −0.087849 | −0.124848 | −0.020658 | −0.029681 | −0.042880 | −0.051828 | −0.074261 | −0.107001 |
| 5 | −0.099193 | −0.148053 | −0.225532 | +0.081774 | +0.105189 | +0.125827 | −0.002153 | −0.011034 | −0.033267 |
| 6 | +0.006727 | +0.011220 | +0.019324 | −0.050052 | −0.071242 | −0.101447 | −0.003932 | −0.005778 | −0.008838 |
| 7 | −0.020862 | −0.029940 | −0.043257 | −0.006600 | −0.009464 | −0.013625 | −0.011963 | −0.017366 | −0.025488 |
| 8 | −0.003574 | −0.006142 | −0.011047 | +0.031962 | +0.045627 | +0.065332 | −0.021568 | −0.032798 | −0.051338 |

| Family | H2+H7 low | H2+H7 high | H2+H10 low | H2+H10 high | H7+H10 low | H7+H10 high |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.051485 | −0.110319 | −0.040052 | −0.084444 | −0.023656 | −0.052534 |
| 2 | +0.010346 | +0.020277 | −0.044735 | −0.092875 | −0.007706 | −0.016777 |
| 3 | −0.139935 | −0.283176 | −0.104125 | −0.228196 | −0.116941 | −0.247831 |
| 4 | −0.082007 | −0.165330 | −0.115470 | −0.239554 | −0.072897 | −0.151557 |
| 5 | −0.013138 | −0.092854 | −0.116461 | −0.342592 | +0.112342 | +0.226878 |
| 6 | −0.043997 | −0.084581 | +0.005458 | +0.019303 | −0.054941 | −0.114328 |
| 7 | −0.027321 | −0.056313 | −0.034179 | −0.074651 | −0.018408 | −0.038478 |
| 8 | +0.028094 | +0.052083 | −0.025352 | −0.064603 | +0.013150 | +0.025563 |

| Family | H2+H7 J low | H2+H7 J high | H2+H10 J low | H2+H10 J high | H7+H10 J low | H7+H10 J high |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.002462 | −0.009447 | +0.000982 | +0.003795 | +0.000075 | +0.000559 |
| 2 | +0.001197 | +0.004731 | −0.000559 | −0.002895 | +0.000938 | +0.003699 |
| 3 | +0.001769 | +0.008125 | −0.003310 | −0.014923 | −0.000781 | −0.003065 |
| 4 | +0.000569 | +0.002399 | −0.001724 | −0.007704 | −0.000412 | −0.001676 |
| 5 | +0.004280 | +0.006851 | −0.015116 | −0.083792 | +0.032721 | +0.134317 |
| 6 | −0.000672 | −0.002458 | +0.002663 | +0.008817 | −0.000957 | −0.004042 |
| 7 | +0.000141 | +0.000570 | −0.001355 | −0.005906 | +0.000155 | +0.000635 |
| 8 | −0.000294 | −0.002202 | −0.000210 | −0.002218 | +0.002756 | +0.011569 |

The high-dose H7+H10 mean interaction includes family 5's +0.134317; the other families include both positive and negative values. The fixed three shuffle orders are not random draws over permutations, so these effects describe this particular order contrast.

## Evidence and checks

The direct CPU process, at source revision `98d14df98ce04ce6b65a4346fe13c5f2cfc8aeef`, exited zero with no recorded interruption or failure. The [receipt](supervisor.json), [manifest](run/manifest.json), [result](run/result.json), [journal](run/attempt.jsonl), [pre-run](pre-run.json) and [execution context](execution-context.json) retain the observed scope and hashes. All 32 raw float32 streams are in `run/`, totaling 167,696,640 bytes. The raw streams retain final-position full-vocabulary logits and the full before/after/projection-output witnesses for no-op and intervention actions. The runner checked finiteness of full-sequence logits, but only final-position logits were saved; an offline replay cannot independently check unsaved positions.

The original attempt's 40 retained files occupy 169,593,335 bytes by file length, including those raw streams, logs, receipt and metadata; the five exported CSVs total another 198,711 bytes. Publication stores another set of the same 167,696,640 raw file-content bytes as Git blobs, and each additional checkout can add that amount uncompressed. Git-server compression, filesystem allocation, audit reports, model files and later copies are not included in those figures. The [direct-process observer](direct-process-observer.py) records the child exit and file hashes without claiming a process-tree attestation.

Independent reviewers replayed all 576 saved action records, all 1,769 journal events, scores, recorded controls, delivered/untouched bits, dose and projection diagnostics, family comparisons and full-vocabulary additive references. They also compared the 56 recorded source hashes to the reviewed source tree. The [numerical replay](independent-numeric-replay.json) and [source/journal replay](independent-source-journal-replay.json) retain full details. These establish saved-data and receipt consistency; they do not independently execute the model, authenticate the receipt or custody reports, establish a complete exposure history, or prove private disjointness. Both custodians reported that no private held-out rows existed for comparison during this attempt.

A subsequent [receipt-bound evidence audit](independent-complete-audit.json) compared the saved identities with separately supplied [source and asset pins](independent-audit-pins.json). When preparing those pins, the reviewer checked fetched source bytes and local checkpoint assets; neither check authenticates the earlier execution. The [reader](https://github.com/EastonSell/model-looking-glass/blob/63f2798d27aa849fbdd3358cdc2931eed90c9400/scripts/audit_copy_pairs_dose_v1.py) was finalized and published after the run; this is a post-run audit binding, not a preregistration. To replay it, first make a separate `pair-dose-attempt` directory containing exactly the `run/` directory and the five original files `pre-run.json`, `execution-context.json`, `supervisor.json`, `stdout.bin` and `stderr.bin` copied verbatim from this result. The reader intentionally rejects the extra report and CSV files in this result directory. Use a checkout whose three reader files match revision `63f2798d27aa849fbdd3358cdc2931eed90c9400` and whose Git object store contains that commit. From the repository root, use a fresh output path:

```sh
python -B scripts/audit_copy_pairs_dose_v1.py --attempt pair-dose-attempt --cases experiments/distilgpt2_copy_head_pairs_dose_v1.json --pins results/distilgpt2-copy-head-pairs-dose-v1-20261028/independent-audit-pins.json --output pair-dose-replay.json
```

- [All family/arm effects and four prompt components](family-arm-effects.csv).
- [All family pair interactions and additive references](family-pair-interactions.csv).
- [All 480 prompt/arm scores, ranks and target top-one statuses](prompt-results.csv).
- [Per-head single deliveries](single-deliveries.csv) and [pair total-dose deliveries](pair-deliveries.csv).
- [Table exporter](export-tables.py), which reads the independent numerical replay.

The projection-output residual between each pair and its two singles ranged from 7.23×10⁻⁷ to 4.46×10⁻⁶ in L2. This is a delivery diagnostic, not a direct interaction test. No inference above depends on a statistical significance or randomization claim.
