# Post-hoc distribution changes in the copying-head ablations

The earlier [copying-head follow-up](../README.md) found that zeroing layer 3
head 2 reduced its repeated-versus-shuffled target log-probability contrast
more than zeroing head 10. Here we reanalyze its **saved full-vocabulary
logits** to ask how much each intervention changed the entire next-token
distribution. No new model forwards were run. This question and these metrics
were chosen after the original results were seen.

For the eight repeated inputs, mean total variation (TV) between the clean
and ablated distributions was **0.18876 for head 2** and **0.09230 for head
10**; head 2 had the larger TV in seven of the eight paired families. For
the 24 shuffled inputs, mean TV was 0.11183 and 0.10690 respectively; head 2
had the larger TV in 13 of the 24 paired inputs. This is a descriptive
comparison of these particular zeroing interventions, not a matched-dose test:
the removed head slices have different norms. TV includes changes to the
target token itself, and the three shuffles within each family are dependent.
The result does not establish a population effect, isolate off-target change,
or identify a unique copying mechanism.

For each saved float32 final-logit vector, the analysis computes float64
softmax probabilities. TV is `0.5 * sum(abs(p_clean - p_ablated))`;
KL is `sum(p_clean * (log(p_clean) - log(p_ablated)))` in nats. Target
probability change is ablated minus clean. `Top changed` counts when the
highest-probability vocabulary token differs from the clean result, regardless
of whether that token is the copy target. Inputs, heads and positions are
those in the frozen original study; `H10 position 5` is the earlier-position
comparison, with all other entries at the final position.

| Intervention | Repeated mean TV | Shuffled mean TV | Repeated mean KL | Shuffled mean KL | Repeated mean target probability change | Top changed repeated / shuffled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H10 final | 0.09230 | 0.10690 | 0.03123 | 0.04807 | -0.04395 | 0/8, 0/24 |
| H2 final | 0.18876 | 0.11183 | 0.17609 | 0.04930 | -0.10449 | 2/8, 1/24 |
| H7 final | 0.08938 | 0.12534 | 0.03568 | 0.05937 | +0.00525 | 0/8, 3/24 |
| H3 final | 0.06472 | 0.05370 | 0.01532 | 0.01053 | +0.03283 | 0/8, 0/24 |
| H1 final | 0.02127 | 0.02263 | 0.00222 | 0.00201 | -0.00877 | 0/8, 1/24 |
| H10 position 5 | 0.00594 | 0.00230 | 0.00025 | 0.00004 | -0.00001 | 0/8, 0/24 |

The eight repeated-input TV comparisons are:

| Family | H10 final TV | H2 final TV |
| ---: | ---: | ---: |
| 1 | 0.03619 | 0.04627 |
| 2 | 0.16186 | 0.26212 |
| 3 | 0.11119 | 0.20024 |
| 4 | 0.11289 | 0.51445 |
| 5 | 0.02864 | 0.06210 |
| 6 | 0.09307 | 0.08673 |
| 7 | 0.10678 | 0.17466 |
| 8 | 0.08776 | 0.16351 |

## Evidence and reproduction

[result.json](result.json) contains all 192 case-by-arm rows, group summaries,
the NumPy version, and the SHA-256 of the original result file
(`4a6c9251a5752714525836df2c29c9e1a8c8659430720a79da47e5f4f3fc18f1`).
[analyze.py](analyze.py) checks the saved byte count and SHA-256 for each
logit segment before calculating the metrics. It reads the prior study's
32 `.f32` files; it does not alter them. From the repository root, with NumPy
installed:

```sh
python results/distilgpt2-copy-head-followup-v2-20261028/posthoc-distribution/analyze.py results/distilgpt2-copy-head-followup-v2-20261028 /tmp/copy-head-distribution.json
```

An independent scalar replay checked the original result SHA, all 224 clean
and intervention logit-segment hashes, and all 192 computed rows. Its largest
absolute numerical difference from the saved metrics was 3.6e-15. This
checks the analysis of saved data; it is not a fresh model execution or proof
of execution provenance. The original study's exposure, intervention,
unequal-norm, and order limitations still apply.
