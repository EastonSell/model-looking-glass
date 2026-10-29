# DistilGPT2 copying: matched-dose head factorial

**Result:** on these eight public exploratory families, the mean joint interaction in target log-probability contrast was −0.024855 nats. Its sign was negative in seven families and positive in one; several were close to zero. The independently calculated hypothetical additive-logit interaction was −0.001884, and the observed joint contrast minus that reference was −0.022971. These are descriptive output-space results, not evidence by themselves of direct H2–H10 cooperation.

The clean repeated target was top-1 on 5 of 8 families; each of H10, H2, H7 and joint attenuation gave 4 of 8. All four interventions lost the same clean top-1 in family 5. The target was top-1 on 0 of 24 shuffled inputs in every condition.

## Design and mean outcomes

The [protocol](https://github.com/EastonSell/model-looking-glass/blob/9dfc2952d3b21c85f0f7afc6bc6d902ed210634c/docs/DISTILGPT2_COPY_HEAD_FACTORIAL_V1.md) fixed the arms before [32 public cases](https://github.com/EastonSell/model-looking-glass/blob/7a5a7d6eaea4c82f02f7f4e05b6e5d41c85ecb49/experiments/distilgpt2_copy_head_factorial_v1.json) were assembled. At zero-based layer 3 and final input position 10, it attenuated the 64-value pre-projection slices for H10, H2, H7, and H2 + H10. The three shuffled variants are fixed permutations, not random samples. The cases are public and nonblind; no population or pretraining-independence claim follows.

For an arm, Δ is target log-probability after attenuation minus clean. The family contrast D is Δ for the repeated input minus the mean Δ for its three shuffled inputs. Negative D means the intervention reduced that particular repeated-versus-shuffled contrast. The following table averages the eight family values; shuffled and repeated components are retained separately.

| Arm | Mean repeated Δ | Mean shuffled Δ | Mean D | Repeated top-1 | Shuffled top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H10 | −0.130265 | −0.001479 | −0.128785 | 4/8 | 0/24 |
| H2 | −0.232165 | −0.053697 | −0.178468 | 4/8 | 0/24 |
| H7 | +0.021791 | −0.026243 | +0.048034 | 4/8 | 0/24 |
| H2 + H10 | −0.396019 | −0.063911 | −0.332109 | 4/8 | 0/24 |

The reported interaction is `J = (D_joint − D_H2) − D_H10` in the protocol’s binary64 order. The additive reference constructs every vocabulary component as `(z_H2 − z_clean) + z_H10` in binary64 from saved logits, without a model forward or float32 re-rounding. Its `J_add` is a hypothetical output-normalization reference, not the exact normalization component of observed J. `D_joint − D_add` can also reflect downstream nonlinearities and rounding; H7 is a single-head comparator, not a two-head interaction null.

The analysis-only [reference clarification](https://github.com/EastonSell/model-looking-glass/blob/7a5a7d6eaea4c82f02f7f4e05b6e5d41c85ecb49/docs/DISTILGPT2_COPY_FACTORIAL_LOGIT_REFERENCE_V1.md) was published after cases were assembled and before the run. It did not alter the interventions, cases, or primary statistic.

| Family | D H10 | D H2 | D H7 | D joint | J | J_add | D joint − D add |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | −0.115400 | −0.227900 | −0.026271 | −0.368220 | −0.024920 | +0.003649 | −0.028569 |
| 2 | −0.232139 | −0.360307 | +0.179934 | −0.634293 | −0.041847 | −0.003949 | −0.037898 |
| 3 | −0.108433 | −0.118648 | −0.008285 | −0.265877 | −0.038796 | −0.002846 | −0.035950 |
| 4 | −0.168083 | −0.299035 | +0.122002 | −0.563572 | −0.096454 | −0.024521 | −0.071933 |
| 5 | −0.115604 | −0.096682 | −0.044711 | −0.220563 | −0.008277 | +0.001516 | −0.009793 |
| 6 | −0.159686 | −0.160039 | +0.049663 | −0.300149 | +0.019576 | +0.013590 | +0.005986 |
| 7 | −0.080491 | −0.102392 | +0.062458 | −0.188880 | −0.005997 | −0.001310 | −0.004688 |
| 8 | −0.050446 | −0.062741 | +0.049483 | −0.115315 | −0.002128 | −0.001205 | −0.000923 |

## Dose and delivery

Within each fixed four-prompt family, `m` was the minimum of the twelve pre-intervention H10, H2 and H7 slice norms. Each single-head intervention requested the same absolute L2 change `m`; the joint arm used the exact same delivered H2 and H10 slices as their single arms, with requested combined L2 √2·m. The smallest slice was fully zeroed. Changing a shuffle can change `m` for the entire family, including its repeated prompt. Equal input-slice L2 does not match the removed fraction, direction, or projection-output effect.

| Family | Requested m | Min requested fraction | Max requested fraction |
| ---: | ---: | ---: | ---: |
| 1 | 2.140102 | 0.339310 | 1.000000 |
| 2 | 1.531761 | 0.230862 | 1.000000 |
| 3 | 1.767630 | 0.242436 | 1.000000 |
| 4 | 2.121997 | 0.315089 | 1.000000 |
| 5 | 1.100445 | 0.188237 | 1.000000 |
| 6 | 1.955713 | 0.368674 | 1.000000 |
| 7 | 1.560399 | 0.225830 | 1.000000 |
| 8 | 1.037202 | 0.164519 | 1.000000 |

The largest observed relative difference between achieved single-slice L2 and `m` was 5.5×10⁻⁸, below the prespecified 10⁻⁵ tolerance. Every targeted 64-value slice changed its intended bits, and single/joint deliveries agreed bit for bit. Full requested and achieved norms, removed fractions, projection-output diagnostics, all four prompt deltas and individual ranks are in [result.json](run/result.json). Clean, no-op and clean-after final logits matched bit for bit; the no-op’s full pre- and post-action inputs also matched. A failed control would make the attempt incomplete; none failed here.

Equal input dose did not imply equal output dose: mean projection-output change norms were 4.555719 (H10), 5.802546 (H2), 5.670833 (H7), and 8.386267 (joint). The per-prompt joint-minus-singles projection-output residual norms averaged 0.00000197; these are descriptive delivery diagnostics, not head-interaction evidence.

## Evidence and checks

The direct CPU attempt used the published [runner](https://github.com/EastonSell/model-looking-glass/blob/5e4a82baed62016950a6f0fd0f6487b3f7d3b698/scripts/copying_head_factorial_v1.py), source tree `1e270b3d87a405d862a7390a621dff799696317d`, and pinned `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`. It recorded 224 starts, 224 retained raw payloads, 224 completions and an observed zero direct-process exit, with no failure record. The 32 `.f32` files contain 61,250,432 bytes. CUDA was hidden. A GPU resource sampler was announced in flight before the factorial start announcement and reaped before the exit announcement; exact process-time overlap was not independently established.

The [manifest](run/manifest.json) records model/checkpoint identity, selected Transformers source-file hashes, library versions and runtime settings. [attempt.jsonl](run/attempt.jsonl), the full [raw streams](run/), [result.json](run/result.json), [stdout](stdout.bin), [stderr](stderr.bin), and the [direct-process receipt](supervisor.json) are retained. The receipt and saved journal are operator/process observations, not independently secured proof of execution or a trusted timestamp. The persistent model-state and buffer hashes matched before and after the run.

The original [pre-run record](pre-run-original.json) incorrectly stated that no separate injection site existed. The independently reviewed [correction](pre-run-correction.json) and [corrected record](pre-run.json) preserve the chronology: a concealed injection site had been drawn, while no injection held-out rows or natural held-out split existed, so no private-row comparison was possible. The [execution context](execution-context.json) records a scoped custody clearance, without revealing the site, and the v12 public reservation preceded the run.

The published [additive reference reader](https://github.com/EastonSell/model-looking-glass/blob/4b15fc7c1ff64b050c7bbe17d0d4d4b6414cf76c/scripts/copy_factorial_logit_reference.py) produced [additive-reference.json](additive-reference.json), including all per-prompt and family values. It checks saved-logit/receipt consistency and does not itself establish journal-event semantics, intervention landing, or authentic execution.

Two independent saved-data replays passed. The [raw audit](independent-raw-audit.py) and [output](independent-raw-audit.json) reconstruct all 713 journal events, 224 logit scores, raw hashes, dose and slice deliveries, unchanged bits, controls and family statistics. The separate [science audit](independent-science-audit.py) and [output](independent-science-audit.json) independently reconstruct the prompt-level additive reference and all family/mean comparisons; its only modification from the reviewer’s original script is resolving inputs relative to the packaged script. These audits verify consistency of retained evidence, not an independent model replay or a unique causal circuit.

Key SHA-256 values:

- `pre-run.json`: `73d6ace2c9ed39d69393af86cb614b93c4d83da2d356d60706890bc49a7efa00`
- `run/manifest.json`: `3fe5f868917b89912b96200ce01029812192c9703c5c39244d6b78485b5605bb`
- `run/attempt.jsonl`: `4e56988a89dddc9aeddcc67c69957e473402140a7a4ed6244cdaa75b5776adfa`
- `run/result.json`: `7f62ed7a014978f71c09ed2e5973d1ce38f544eff0f2a1652cc9fd75e2fb125b`
- `supervisor.json`: `35626526739498d468c9e91d1c4f39e3f79603cd9015fc7fbd386ba3917f086d`
- `additive-reference.json`: `be189d8ac0fae4b84f1971a8bcc0b81980a28942b158fa61c2450a88f18fb30f`
