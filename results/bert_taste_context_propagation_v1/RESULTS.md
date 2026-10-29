# Taste-context propagation: matched interventions on six public stems

An intervention at the block-0 ingredient position produced a donor-aligned
odd response at each measured block-3 position on these cases. The
prespecified contrast against two matched orthogonal directions was positive
in all 24 recipient/template cells at each later position. A direct block-3
`the` intervention moved the sweet–sour score opposite its lexical donor in
all 24 cells. These observations are compatible with a propagated contextual
response and a separately active, position-sensitive output effect; they do
not identify a unique route or a general taste representation.

The six stems are orange, cereal, soda, toast, bread and apple, crossed with
both public templates and both ingredients. All cases enter the summaries.
They are correlated convenience examples; the templates and block 3 were
chosen after earlier results. Prior unknown exposure and pretraining exposure
are not ruled out. The study does not treat the 24 cells as independent
trials or run a significance test.

## Early ingredient to later positions

For each recipient, the propagation contrast is the projection of the odd
response onto that position's clean donor-minus-recipient direction, minus
the mean absolute projections from the two fixed orthogonal controls. Each
table value averages both directions and six stems; difference is `after`
minus `with`.

| Block-3 position | `after` | `with` | Paired difference |
|---|---:|---:|---:|
| Ingredient | +0.045860 | +0.046957 | -0.001097 |
| `the` | +0.027479 | +0.029126 | -0.001648 |
| `tasted` | +0.021377 | +0.024301 | -0.002923 |
| `[MASK]` | +0.018136 | +0.027441 | -0.009305 |

The same-ingredient cross-template direction is an active control. In a
post-hoc comparison, its absolute projection exceeds the lexical projection
in two of 12 `with`/`tasted` cells and one of 12 `after`/mask cells. These
are descriptive control comparisons, not an additional prespecified test.

## Direct taste-score comparisons

The target is the `[MASK]` sweet logit minus sour logit. The orthogonal
contrast is donor-signed lexical odd effect minus the mean absolute odd
effect of the two orthogonal axes. The context contrast substitutes the
absolute same-ingredient cross-template odd effect. Equal input norm does
not equalize position sensitivity.

| Intervention site | Orthogonal `after` | Orthogonal `with` | Context `after` | Context `with` |
|---|---:|---:|---:|---:|
| Block 0 ingredient | +0.033609 | +0.027748 | +0.026800 | +0.027774 |
| Block 3 ingredient | +0.037043 | +0.038547 | +0.050962 | +0.041013 |
| Block 3 `the` | -0.037327 | -0.025239 | -0.032674 | -0.038878 |
| Block 3 `tasted` | +0.094077 | +0.023690 | +0.007846 | +0.041259 |

The direct `the` donor-signed lexical odd means are −0.021433 and −0.015097;
both controls' contrasts are negative in all 24 cells. At block-3 `tasted`,
the orthogonal contrast is positive in 21/24 cells, while the context
contrast is negative in 5/24. At block-0 ingredient, the orthogonal contrast
is negative in 4/24 and the context contrast in 2/24. These negative and
ineligible cases remain in the [complete case table](tables/all-case-axis.csv).

Whole-vocabulary total variation also shows active controls. For direct
block-3 `tasted`, mean lexical `+`/`−` TV is 0.042895/0.042593 for `after`
and 0.019815/0.019801 for `with`. Pooling the two orthogonal axes and signs
gives 0.02302 and 0.02124; the cross-template `+`/`−` control gives
0.033562/0.034737 and 0.028478/0.027648. Nonzero output shifts do not by
themselves identify whether an activation carries a particular concept.

The six common radii were 1.104221–1.464037; realized displacements were
5.65%–8.12% of recipient activation norm, with maximum relative dose error
6.93×10⁻⁸. All 96 self patches, 24 fresh checks and 24 final-block donor
patches passed the owner's exact clean-control checks. Their full logits were
not retained; the saved audit checks their matching digests and selected
summaries. All 768 non-self source and landing vectors were independently
reconstructed from saved bits.

## Clean baselines

The strict earlier sweet/sour margin and top-five eligibility rule holds for
10 of 12 stem/template pairs: 6/6 `after`, 4/6 `with`. The `with` lemon
baselines for cereal and orange are still sweet-favored. Eligibility was
descriptive and did not filter the results. The probabilities below are
full-vocabulary softmax probabilities; a two-token margin is not the
probability that the cloze is correct.

| Public row | Sweet − sour | P(sweet) | P(sour) | Top five tokens |
|---|---:|---:|---:|---|
| apple-after-lemon | -0.8184 | 0.0306 | 0.0693 | lemon, sour, bitter, better, orange |
| apple-after-sugar | +0.2302 | 0.0720 | 0.0572 | better, sweet, good, bitter, sour |
| apple-with-lemon | -0.1405 | 0.0496 | 0.0571 | sour, sweet, bitter, orange, better |
| apple-with-sugar | +0.7620 | 0.0776 | 0.0362 | sweet, good, bitter, sour, delicious |
| bread-after-lemon | -0.2236 | 0.0661 | 0.0827 | sour, sweet, good, bitter, lemon |
| bread-after-sugar | +0.7665 | 0.1052 | 0.0489 | sweet, good, better, delicious, sour |
| bread-with-lemon | -0.0694 | 0.0731 | 0.0784 | sour, sweet, salty, bitter, delicious |
| bread-with-sugar | +0.9071 | 0.1056 | 0.0426 | sweet, good, salty, delicious, sour |
| cereal-after-lemon | -0.4268 | 0.0502 | 0.0769 | better, good, sour, lemon, bitter |
| cereal-after-sugar | +0.6929 | 0.0807 | 0.0404 | good, better, sweet, sour, excellent |
| cereal-with-lemon | +0.0280 | 0.0731 | 0.0711 | sweet, sour, good, bitter, better |
| cereal-with-sugar | +1.0931 | 0.0926 | 0.0311 | good, sweet, better, salty, sour |
| orange-after-lemon | -0.4943 | 0.0380 | 0.0623 | better, bitter, sour, good, lemon |
| orange-after-sugar | +0.3667 | 0.0714 | 0.0495 | better, sweet, good, bitter, delicious |
| orange-with-lemon | +0.0182 | 0.0455 | 0.0447 | bitter, mixed, sweet, sour, pleasantly |
| orange-with-sugar | +0.8157 | 0.0849 | 0.0376 | sweet, bitter, delicious, good, sour |
| soda-after-lemon | -0.7706 | 0.0402 | 0.0869 | sour, good, better, lemon, bitter |
| soda-after-sugar | +0.1618 | 0.0720 | 0.0613 | good, better, sweet, sour, bitter |
| soda-with-lemon | -0.2848 | 0.0548 | 0.0728 | sour, sweet, bitter, good, mixed |
| soda-with-sugar | +0.6559 | 0.0883 | 0.0458 | sweet, good, sour, bitter, rotten |
| toast-after-lemon | -0.7496 | 0.0368 | 0.0779 | sour, delicious, good, bitter, better |
| toast-after-sugar | +0.0904 | 0.0534 | 0.0488 | good, delicious, better, sweet, sour |
| toast-with-lemon | -0.4126 | 0.0441 | 0.0667 | sour, sweet, delicious, good, bitter |
| toast-with-sugar | +0.3898 | 0.0568 | 0.0385 | good, sweet, delicious, sour, rotten |

## Evidence and limits

The frozen [protocol](../../docs/BERT_TASTE_CONTEXT_PROPAGATION_V1.md) and
[public reservation](../../experiments/bert_taste_propagation_public_cases_v1.json)
are bound to source [900bf64e](https://github.com/EastonSell/model-looking-glass/commit/900bf64e17e5a3b2f3a0352a207d27c107baa91f);
the recorded run source is [a27b3453](https://github.com/EastonSell/model-looking-glass/commit/a27b34533c3da5f34a541ef2b8b57631bf42e819).
The selector's public source was independently read before the recorded
tokenizer-only selection; operator timestamps and process records are
cooperative observations, not a trusted chronology or execution attestation.
The [observed exposure successor](../../experiments/bert_taste_propagation_public_exposures_v2.json)
preserves the reservation rows and result/source hashes.

The [result](run/result.json), [started record](run/started.json) and six
complete journals ([orange](run/orange.jsonl), [cereal](run/cereal.jsonl),
[soda](run/soda.jsonl), [toast](run/toast.jsonl), [bread](run/bread.jsonl),
[apple](run/apple.jsonl)) retain every input ID/mask, source/landing and
downstream selected vector, and all 792 baseline/intervention full mask-logit
vectors. [All case/axis values](tables/all-case-axis.csv), [all downstream
axis values](tables/all-downstream-axis.csv), and [all aggregates](tables/summary.json)
retain both signs and every negative or null result.
The [file inventory](INVENTORY.json) records sizes and SHA-256 values for
the result package; it excludes itself.

The [independent Fraction-based reader](audit/audit_bert_context_propagation.py)
reconstructed the schedule, doses, full-logit metrics, selected vectors and
aggregates; its [report](audit/bert-propagation-independent-audit-v1.json) and
[synthetic corruption tests](audit/test_audit_bert_context_propagation.py)
are included. Its sibling helper files and toy fixture are required. A
[second saved-vector replay](audit/review_taste_propagation_outcomes.py) and
[report](audit/taste_propagation_independent_outcomes.json) independently
check full logits, token labels and the main comparisons using the pinned
tokenizer. The [process audit](audit/bert_propagation_process_audit.json)
checked the original private records and journals; its included source
requires that original workspace and those private records. The public
[spawn](receipts/operator-spawn-public.json), [exit](receipts/operator-exit-public.json)
and [outer observer](receipts/outer-observation-public.json) projections
remove process IDs and bind original record hashes; those projections alone
cannot authenticate the originals. A zero direct-process exit and an empty
prefix were recorded; whole-tree cleanup is not attested.

Untouched whole-activation coordinates were checked in process, but those
full tensors were not retained; the saved selected slices cannot independently
establish that whole-tensor witness. The audits do not verify execution or
unknown training exposure. Their exact binary64/libm reductions may require a
compatible runtime. Run the portable reader with `python -S -B
audit/audit_bert_context_propagation.py --manifest
../../experiments/bert_taste_propagation_public_cases_v1.json --run-dir run
--output <new-report.json>` from this directory; its synthetic test can be run
with `python -S -B audit/test_audit_bert_context_propagation.py`.
