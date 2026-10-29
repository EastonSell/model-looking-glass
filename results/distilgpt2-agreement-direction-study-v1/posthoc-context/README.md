# Post-hoc analysis: context response and vocabulary shifts

**This is a post-hoc, model-free analysis of saved logits, not another model run or a new confirmatory test.** The original [fixed study and its findings](../RESULTS.md) are unchanged: its two exploratory criteria passed on the published, nonblind cases. The following comparisons were chosen after seeing those results. They describe this small set of related prompts; they do not establish a grammatical mechanism, a unique circuit, or an input-independent output bias.

The positive `u` intervention increased `P(is)` and decreased `P(are)` in all eight subjectless fragments. Its signed effect also extended to `was − were`, and the full probability distributions changed beyond `is` and `are`. A shared number-related output component is consistent with these observations, but an identical input-independent `is − are` shift at a given requested dose is too simple for the target pairs: the singular member had the larger dose-normalized signed response in all 16 same-lemma pairs. A shared component with input interactions, lexical or attractor effects, and nonlinear response all remain viable. These correlated pairs are descriptive, not 16 independent tests.

## Response per requested input dose

For a token-pair margin `m`, we calculate `O = (m(+q) − m(−q))/2` and report `O/q`. Here `q` is the original requested L3 attention-output L2 dose, not an output change. This finite-response normalization is not a derivative and does not create an equal-dose causal comparison across prompts. Each same-lemma singular/plural target pair has the same `q`; a context's `q` is the average of its family's two target-pair doses. All doses were positive.

| Cases | n | Mean `u` O/q: is − are | Mean `u` O/q: was − were | Range of `u` O/q: is − are |
| --- | ---: | ---: | ---: | ---: |
| Context fragments | 8 | 0.2512 | 0.2316 | 0.1864–0.3156 |
| Singular-subject targets | 16 | 0.2411 | 0.2161 | 0.1692–0.3607 |
| Plural-subject targets | 16 | 0.0998 | 0.1011 | 0.0690–0.1606 |

For the eight contexts, mean signed `is − are` O/q was −0.0460 for `v1` and −0.0081 for `v2`, compared with +0.2512 for `u`. The mean `u` O/q was +0.0046 for `the − a` and +0.0020 for `in − on`; these are two selected unrelated comparisons, not a general null result. `v1` is a fixed number-by-lemma contrast and `v2` is one fixed pseudorandom orthogonal direction. Neither is a randomization distribution.

The even component `E = (m(+q) + m(−q))/2 − m(clean)` is also saved. For `is − are` it ranges from −0.3237 to +0.4831 across the eight contexts, so plus and minus shifts should not be treated as necessarily symmetric around clean.

## Eight context fragments

Total variation is half the sum of absolute next-token probability changes between clean and `u+`. The last column is the share of the **absolute probability-change L1 sum** attributable to the two `is`/`are` token IDs, not a share of model knowledge or a claim about other contexts. Percentages below are descriptive.

| Context | q | `u` is − are O/q | `u` was − were O/q | TV: clean vs `u+` | `is`/`are` share of L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Near the fence | 3.5961 | 0.2943 | 0.2881 | 2.24% | 4.93% |
| Near the fences | 3.6645 | 0.2222 | 0.1926 | 4.49% | 13.03% |
| Beside the tree | 2.5438 | 0.2094 | 0.2041 | 1.59% | 2.36% |
| Beside the trees | 2.8303 | 0.1864 | 0.1593 | 3.19% | 18.19% |
| Behind the window | 3.4212 | 0.2600 | 0.2473 | 2.01% | 7.35% |
| Behind the windows | 3.0895 | 0.3156 | 0.2890 | 3.91% | 26.65% |
| Outside the door | 4.4027 | 0.2172 | 0.1986 | 3.77% | 4.23% |
| Outside the doors | 4.1446 | 0.3044 | 0.2740 | 8.58% | 21.03% |

For example, after `Outside the doors`, `P(are)` went from 8.015% to 5.276%, while `P(is)` went from 0.314% to 1.184%. The largest increase was instead the single token `" of"`, by about 1.93 percentage points. Across the eight contexts, the two agreement tokens account for 2.36%–26.65% of the absolute probability change (mean 12.22%). These results are consistent with a broader output-distribution shift and do not identify its cause. The corresponding `u−` TV range is 1.99%–4.63%.

Every fragment retains a number-marked attractor; none is a neutral no-number context. Two `Beside` fragments have five tokens and the other six have three. The cases share four frames, were public and nonblind, and were not the separately sealed natural held-out split. The same finite set motivated this analysis, so these numbers carry no inferential or out-of-sample guarantee.

## Full-vocabulary response geometry

For each case and each direction, we form the 50,257-dimensional vector `(logits(+q) − logits(−q))/(2q)`, subtract its own mean across token coordinates, then compute pairwise cosine similarities. Removing a uniform logit offset makes the comparison invariant to a change that would leave softmax probabilities unchanged. The cosine equally weights all vocabulary coordinates, including very low-probability tokens; it is not a probability-weighted similarity or a mechanistic test. Dividing by the positive scalar `q` does not itself alter cosine.

| Case pairs | n | Mean cosine: `u` | Mean cosine: `v1` | Mean cosine: `v2` |
| --- | ---: | ---: | ---: | ---: |
| Context–context | 28 | 0.654 | 0.755 | 0.781 |
| Target–context | 256 | 0.553 | 0.623 | 0.682 |
| Target–target | 496 | 0.605 | 0.773 | 0.820 |

The positive shared geometry is not unique to `u`: both fixed controls have larger means by this measure. Similarity and a shared `is`/`are` direction therefore do not by themselves distinguish a context-independent decoder bias from input-dependent effects. The full pair values are in [pairwise-vocabulary-cosines.csv](pairwise-vocabulary-cosines.csv); [per-case.csv](per-case.csv) records all 40 cases by three directions, four margins, plus/minus probabilities and TV, the even component, and the individual tokens with largest probability increases and decreases. Token strings are decoded single-token labels, not generated continuations.

## What a prospective follow-up could distinguish

A new, separately frozen design could include genuinely no-number fragments, and cross explicit singular/plural subjects with singular/plural attractors in matched templates. It could balance lexical forms and token lengths, include new held-out frames, and prespecify the comparison before choosing or evaluating new cases. A fixed dose grid and behavioral-matched controls would help assess input dependence and nonlinearity; reporting only `O/q` does not establish either. A prespecified additive output-logit benchmark could test how much of the response a common bias explains and where it fails. Testing other singular/plural auxiliaries alongside unrelated token pairs and the full-vocabulary distributions would help distinguish a narrow `is`/`are` effect from a more general output shift. Such a study would still need its own custody, exposure, and uncertainty controls.

## Reproduction and scope

The [script](../../../tools/agreement_direction_context_posthoc.py) reads the original 809-event journal by exact SHA-256 and the content-addressed payloads. It validates the ordered 784-forward inventory; for the selected analyses it recomputes all eight family geometries, checks clean/no-op/clean-after full-logit equality, and checks all 240 signed intervention plans, sites, landed values, doses and untouched hook slices. It reads 400 clean/repeat/no-op/selected logit events; no nonzero float32 subnormal values occurred in those events. IEEE float32 bits are decoded explicitly into float64, and calculations use a shifted softmax. The pinned cases, direction vectors and tokenizer are listed in [summary.json](summary.json). The original independent replay checks the complete study, including other arms; this analysis checks the subset it consumes. Recorded process, custody, exposure and full-sequence-finiteness statements remain observations, not authentication of execution. Only last-position logits were saved.

Rebuild and extract the original archive as described in the [study results](../RESULTS.md), then from a checkout containing the script run Python with NumPy and `tokenizers` installed:

```text
python -B -m tools.agreement_direction_context_posthoc \
  --repo-root . \
  --journal-directory <extracted-evidence>/run/attempt \
  --tokenizer-json <path-to-distilgpt2-tokenizer.json> \
  --output-directory <new-empty-path>
```

The tokenizer file must hash to `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`; the output directory must not already exist. Reproductions on different numerical-library versions can vary in the final displayed digits. The tables and CSVs preserve all 40 cases and the complete pair denominators, rather than selecting only large effects. The rounded values above can be checked against the accompanying full-precision outputs and the original independent [science replay](../science-replay.json).
