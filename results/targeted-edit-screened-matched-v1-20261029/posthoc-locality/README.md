# Post-hoc locality and efficacy sensitivity

This is a **post-hoc, model-free analysis** of the [screened study's saved
logits](../README.md). No new model or edited weight was run. It does not
change the original result: both prespecified comparisons were negative because
the projected edit reversed only one of eight target margins and changed one of
12 locality top-token IDs. The cases were public, and the eight targets were
selected by their baseline margins; they are not a blind or representative
sample.

## What the actual saved distributions show

The projected edit had a smaller `KL(original || edited)` than the
discovery-matched unprojected edit on all 12 locality prompts. Their mean KLs
were **0.000277** and **0.000430** nats, respectively. Their mean total
variation distances were **0.00587** and **0.00801**. Total variation is half
the sum of the absolute probability differences across all 50,257 next-token
choices: 0.00587 corresponds to about 0.587 percentage points of probability
mass redistributed. These summaries concern the 12 listed next-token
distributions, not preservation of a general capability.

The target effects were unequal. Mean London-minus-Paris margin gain over the
eight selected targets was **0.717697** for projected and **0.805818** for
the discovery-matched unprojected edit. The latter was matched on the older
discovery prompt, not on these eight targets. Projected gain was smaller on
seven targets; the projected/matched ratio ranged from **0.073 to 1.036**.
For “When asked to name the capital of France, answer,” the gains were
0.022102 versus 0.302292. For “The French Republic's capital city is,” they
were 1.238190 versus 1.195625. The [target comparison](target-comparison.csv)
contains all eight rows. Matching a mean cannot make these effects equal for
each prompt. Neither London nor Paris was the top token for any of the eight
targets in any saved condition; a pairwise margin reversal does not establish
that a generated answer became London.

The equal-weight-norm unprojected edit had mean target gain 1.074435, mean
locality KL 0.000768 and mean total variation 0.01069. The nuisance control
was similar on these aggregate quantities; the projected-random and nuisance
controls each use one fixed draw. None supplies a sampling distribution or a
statistical significance claim.

## An efficacy sensitivity calculation

To show how the locality comparison depends on that mismatch, this analysis
forms a **synthetic logit interpolation**, for every saved row:

`z_interpolated = z_original + 0.8906440652 * (z_matched - z_original)`.

The factor is the ratio of the two mean gains on the **same eight evaluation
targets**. It therefore makes the interpolated mean target-margin gain equal
to the projected mean by construction. It does not match each target: the
per-target interpolated and projected gains remain different in the
[comparison file](target-comparison.csv).

The interpolated distributions have mean locality KL **0.000340** and mean
total variation **0.00713**. Projected KL and total variation are each lower
on **8 of 12** locality prompts relative to this interpolation, rather than
12 of 12 relative to the actual matched-control logits. This is a sensitivity
view of these particular saved distributions. It is **not** a model forward,
an evaluated weight dose, or evidence that the model would respond linearly
to a smaller edit. The factor was chosen after seeing the evaluation targets;
this calculation cannot establish an efficacy-matched causal comparison.

## Locality by prompt

The table reports the probability of the specified continuation at baseline,
the projected edit's change in that probability in **percentage points**,
and projected total variation. The three KL columns are multiplied by 1,000
for legibility. “Interpolated” is the synthetic calculation above; the other
two columns are measured saved logits. Expected continuations were the
baseline top token on only 3 of these 12 prompts.

| Expected token | Prompt | Baseline probability | Projected change, pp | Projected TV | Projected KL × 1,000 | Actual matched KL × 1,000 | Interpolated KL × 1,000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| London | The national capital of the United Kingdom is | 0.3073% | +0.153262 | 1.128% | 0.8928 | 1.1816 | 0.9295 |
| Lisbon | The national capital of Portugal is | 0.0229% | -0.001791 | 1.133% | 0.5756 | 0.6542 | 0.5186 |
| Warsaw | The national capital of Poland is | 0.0929% | -0.009374 | 1.175% | 0.5780 | 0.6187 | 0.4906 |
| Stockholm | The national capital of Sweden is | 0.0107% | -0.000249 | 1.082% | 0.4811 | 0.5488 | 0.4351 |
| Paris | The Louvre Museum, formerly a royal palace, is in | 1.0127% | -0.177522 | 0.576% | 0.4222 | 0.5810 | 0.4611 |
| French | France's official language is | 1.3630% | -0.040930 | 0.186% | 0.0206 | 0.1481 | 0.1175 |
| euro | Metropolitan France uses the currency called the | 8.7133% | -0.029487 | 0.353% | 0.0527 | 0.0615 | 0.0488 |
| Europe | Metropolitan France lies on the continent of | 3.3638% | +0.030558 | 0.906% | 0.2639 | 1.1762 | 0.9343 |
| three | The number of sides on a triangle is | 0.2005% | +0.000624 | 0.129% | 0.0063 | 0.0620 | 0.0492 |
| oxygen | Water consists chemically of hydrogen and | 36.3073% | -0.052096 | 0.138% | 0.0082 | 0.0161 | 0.0128 |
| minutes | A full hour consists of sixty | 43.7983% | -0.011633 | 0.018% | 0.0001 | 0.0099 | 0.0078 |
| fourteen | The number of days in a fortnight is | 0.0017% | -0.000016 | 0.221% | 0.0191 | 0.0970 | 0.0769 |

On the Louvre prompt, projected `P( Paris)` fell from 1.013% to 0.835%,
while `P( London)` rose by 0.220 percentage points. On the four other-country
capital prompts, the largest individual probability increase was for the
token ` home`, by 0.42–0.58 percentage points. Thus the full distribution
change is not confined to London and Paris. On the currency prompt, the top
token changed from ` euro` to ` Euro`; this fails the original exact-token
criterion, but the capitalization change alone does not establish lost
currency knowledge. A token's displayed text is its single decoded GPT-2
token, not a generated answer.

The [locality CSV](locality.csv) has all 72 locality/condition rows, including
KL, total variation, Jensen-Shannon divergence, expected-token rank and
probability, top-token IDs and the largest positive and negative probability
shifts. The [target CSV](targets.csv) has all 48 target/condition rows; the
[interpolated-locality CSV](interpolated-locality.csv) contains all 12
synthetic comparisons. The [summary](summary.json) records fixed input hashes
and aggregate values. These are descriptive diagnostics; the limited prompts,
selection, unknown training exposure, and recorded rather than authenticated
execution prevent broad preservation or statistical-certainty claims.

## Reproduce

From the repository root, with NumPy 2.4.6 and the pinned DistilGPT-2
`tokenizer.json` (SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`):

```text
python -B tools/screened_locality_posthoc.py \
  --tokenizer-json PATH_TO_TOKENIZER_JSON \
  --output-dir NEW_OUTPUT_DIRECTORY
```

The output directory must not exist. The script checks the fixed case,
selection and phase hashes, all seven matrix hashes and every row's saved
identity and SHA-256 before computing stable binary64 softmax, KL, total
variation and Jensen-Shannon divergence. It checks the tokenizer bytes before
decoding token IDs. Library/platform rounding can affect the last displayed
digits. The published process and scientific audits remain separate evidence
about the original attempt and its limitations.
