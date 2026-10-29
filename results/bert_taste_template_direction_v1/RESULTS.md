# Crossed-template taste-context follow-up: public CPU result

The fixed public follow-up completed 336 recorded forwards on the pinned CPU DistilBERT checkpoint. Across the six stems, the mean donor-signed ingredient-versus-`the` toward contrast was +0.084965 in `after adding` and +0.067577 in `with some`; the paired `after` minus `with` mean was +0.017388. Five stem differences were positive; syrup was slightly negative (−0.000061). These are descriptive, correlated observations on six public convenience stems, not independent trials or a population claim.

The `the` comparison was active: its donor-signed toward response was negative in all 24 stem/template/direction cells, with means −0.022960 (`after`) and −0.014753 (`with`). The corresponding ingredient means were +0.062005 and +0.052824. Subtracting the active `the` response contributes to the positive position contrast; it is not a no-effect null.

The donor-signed odd ingredient response minus the absolute odd orthogonal response averaged +0.040383 (`after`) and +0.036340 (`with`), with paired mean +0.004043. Chocolate, syrup and cream had negative paired differences for this quantity. Mean absolute orthogonal odd responses were 0.018684 and 0.016776 respectively; a single artificial orthogonal direction is not a generic null, and both ± perturbations may be off-manifold.

The prespecified direction main contrast was +0.011770 and the template-by-direction interaction was +0.016392. Block 3 was chosen after the earlier study. No significance or general taste-concept claim is made.

Clean-answer eligibility was descriptive only: 4/6 `after` pairs and 3/6 `with` pairs met the earlier rule, with only syrup and cream eligible in both templates. All 12 pairs and both directions enter the primary analysis. Full baseline logits and target probabilities are retained; a separate tokenizer-only review checked the recorded top-token labels against their IDs.

The common requested radius per stem was 1.009546–1.447405; in every stem it was set by the `after` contextual-`the` distance. Actual ingredient/orthogonal displacements were 5.23%–7.69% of recipient norm and `the` displacements 5.62%–8.15%; maximum relative dose error was 4.50×10⁻⁸. All 48 self patches and 24 final-block donor controls had zero saved target-margin effects. Raw achieved doses and cosines are in [delivered-doses.csv](delivered-doses.csv).

## Per-stem summaries

| Stem | After | With | After − with | Direction main | Template × direction | Odd-specificity after | Odd-specificity with | Specificity difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chocolate | +0.071352 | +0.048443 | +0.022909 | +0.020941 | +0.021460 | +0.041815 | +0.042669 | -0.000854 |
| rice | +0.099812 | +0.078818 | +0.020994 | +0.010725 | +0.026858 | +0.072679 | +0.026234 | +0.046445 |
| salad | +0.093184 | +0.077221 | +0.015963 | +0.001722 | +0.008246 | +0.028424 | +0.017062 | +0.011363 |
| syrup | +0.067925 | +0.067986 | -0.000061 | +0.014175 | +0.017515 | +0.027693 | +0.035119 | -0.007426 |
| dessert | +0.076642 | +0.059194 | +0.017448 | +0.001390 | +0.004062 | +0.051972 | +0.040546 | +0.011426 |
| cream | +0.100875 | +0.073801 | +0.027074 | +0.021668 | +0.020213 | +0.019712 | +0.056408 | -0.036695 |

## Every template and direction

The first contrast is signed ingredient-toward minus signed `the`-toward. The second is signed ingredient odd response minus absolute orthogonal odd response. Eligibility is a descriptive pair label, not a filter.

| Stem | Template | Direction | Eligible | Position contrast | Odd specificity |
|---|---|---|---|---:|---:|
| chocolate | after | sugar_to_lemon | yes | +0.087188 | +0.063861 |
| chocolate | after | lemon_to_sugar | yes | +0.055517 | +0.019769 |
| chocolate | with | sugar_to_lemon | no | +0.053549 | +0.049871 |
| chocolate | with | lemon_to_sugar | no | +0.043338 | +0.035467 |
| rice | after | sugar_to_lemon | yes | +0.111889 | +0.084410 |
| rice | after | lemon_to_sugar | yes | +0.087734 | +0.060948 |
| rice | with | sugar_to_lemon | no | +0.077466 | +0.031557 |
| rice | with | lemon_to_sugar | no | +0.080170 | +0.020911 |
| salad | after | sugar_to_lemon | no | +0.096107 | +0.027390 |
| salad | after | lemon_to_sugar | no | +0.090261 | +0.029459 |
| salad | with | sugar_to_lemon | yes | +0.076020 | +0.021954 |
| salad | with | lemon_to_sugar | yes | +0.078422 | +0.012170 |
| syrup | after | sugar_to_lemon | yes | +0.079391 | +0.034809 |
| syrup | after | lemon_to_sugar | yes | +0.056458 | +0.020577 |
| syrup | with | sugar_to_lemon | yes | +0.070695 | +0.039043 |
| syrup | with | lemon_to_sugar | yes | +0.065277 | +0.031194 |
| dessert | after | sugar_to_lemon | no | +0.078352 | +0.056899 |
| dessert | after | lemon_to_sugar | no | +0.074931 | +0.047045 |
| dessert | with | sugar_to_lemon | no | +0.058873 | +0.045733 |
| dessert | with | lemon_to_sugar | no | +0.059514 | +0.035358 |
| cream | after | sugar_to_lemon | yes | +0.116762 | +0.034873 |
| cream | after | lemon_to_sugar | yes | +0.084988 | +0.004552 |
| cream | with | sugar_to_lemon | yes | +0.079581 | +0.069430 |
| cream | with | lemon_to_sugar | yes | +0.068020 | +0.043385 |

## All clean baselines

Probabilities are over the full vocabulary; a sweet–sour margin is only their log-odds difference. Baseline rankings can be reconstructed from the saved full logits.

| Stem | Template | Ingredient | Sweet − sour | P(sweet) | P(sour) | Recorded top five |
|---|---|---|---:|---:|---:|---|
| chocolate | after | sugar | +0.629004 | 0.068092 | 0.036301 | good, sweet, delicious, better, perfect |
| chocolate | after | lemon | -0.225739 | 0.046371 | 0.058114 | good, delicious, better, sour, bitter |
| chocolate | with | sugar | +1.039170 | 0.083677 | 0.029601 | sweet, good, delicious, salty, sour |
| chocolate | with | lemon | +0.120184 | 0.055563 | 0.049271 | sweet, sour, delicious, mixed, good |
| rice | after | sugar | +0.676126 | 0.126349 | 0.064259 | sweet, sour, better, bitter, salty |
| rice | after | lemon | -0.475915 | 0.057621 | 0.092740 | sour, lemon, bitter, sweet, better |
| rice | with | sugar | +1.333367 | 0.125421 | 0.033060 | sweet, salty, good, sour, bitter |
| rice | with | lemon | +0.209682 | 0.070041 | 0.056792 | sweet, sour, salty, bitter, mixed |
| salad | after | sugar | -0.006090 | 0.054061 | 0.054392 | good, delicious, better, sour, sweet |
| salad | after | lemon | -0.812706 | 0.032153 | 0.072473 | sour, good, lemon, better, delicious |
| salad | with | sugar | +0.639841 | 0.065884 | 0.034746 | delicious, good, sweet, salty, sour |
| salad | with | lemon | -0.188613 | 0.046459 | 0.056103 | sour, delicious, sweet, good, bitter |
| syrup | after | sugar | +0.443610 | 0.093916 | 0.060267 | sweet, good, bitter, sour, better |
| syrup | after | lemon | -0.379082 | 0.065927 | 0.096316 | sour, bitter, sweet, good, better |
| syrup | with | sugar | +0.855997 | 0.111667 | 0.047443 | sweet, good, bitter, sour, pleasant |
| syrup | with | lemon | -0.015450 | 0.076486 | 0.077677 | sour, sweet, bitter, pleasant, good |
| dessert | after | sugar | +0.977676 | 0.083293 | 0.031333 | delicious, sweet, good, perfect, better |
| dessert | after | lemon | +0.044079 | 0.054501 | 0.052150 | delicious, lemon, sweet, sour, good |
| dessert | with | sugar | +1.727340 | 0.114274 | 0.020313 | sweet, delicious, good, heavenly, vanilla |
| dessert | with | lemon | +0.641223 | 0.082477 | 0.043436 | sweet, delicious, sour, good, bitter |
| cream | after | sugar | +0.572896 | 0.062638 | 0.035321 | good, better, sweet, delicious, sour |
| cream | after | lemon | -0.329762 | 0.047081 | 0.065473 | good, better, sour, bitter, sweet |
| cream | with | sugar | +0.909970 | 0.082567 | 0.033236 | good, sweet, delicious, better, sour |
| cream | with | lemon | -0.143477 | 0.056726 | 0.065478 | sour, sweet, bitter, mixed, good |

## Evidence and scope

The [fixed protocol](../../docs/BERT_TASTE_TEMPLATE_DIRECTION_V1.md), [24 public cases](../../experiments/bert_taste_template_direction_public_cases_v1.json), [selection observation](../../experiments/bert_taste_template_direction_selection_v1.json), and [observed exposure successor](../../experiments/bert_taste_template_direction_public_exposures_v2.json) bind the public selection and the saved result. The exact [selection source](https://github.com/EastonSell/model-looking-glass/tree/210dfc676f9e1670391f8fe71f2a9c418d6fe962) and [run source](https://github.com/EastonSell/model-looking-glass/tree/05e55bffacbf5019d181546286c2743b8ccbf487) are separately pinned. The operator observed the source publication before tokenizer-only selection; no independently trusted exact-head timestamp was retained. The cases are nonblind and unknown previous or pretraining exposure is not excluded.

The [all-case table](all-cases.csv) contains each of 24 directional cells, selected-logit bits, matched effects and odd/even responses. [All actions](all-actions.csv), [delivered doses](delivered-doses.csv), [result](run/result.json), [start record](run/started.json) and six per-stem journals retain the full reported outcomes. The separate [independent reader](audit/audit_bert_template_direction.py) and [report](audit/independent-audit.json) replay the 336 saved forwards and reconstruct matched geometry from bits; they do not run the model. The reader requires its sibling `bert_taste_matched_independent.py` and an output path that does not yet exist. A [second independent reader](audit/independent-contrasts.py) and [report](audit/independent-contrasts-report.json) reconcile the selected-bit effects, contrasts and baseline rankings; the latter also checks saved token labels against the pinned tokenizer.

From this results directory, the first replay is `python -I -S -B audit/audit_bert_template_direction.py ../../experiments/bert_taste_template_direction_public_cases_v1.json run <new-report.json>`. Its fabricated tests run with `python -S -B audit/test_audit_bert_template_direction.py`. The second reader accepts `--run run --cases ../../experiments/bert_taste_template_direction_public_cases_v1.json` and an optional `--tokenizer-json` for the pinned tokenizer; use ordinary Python without `-O` because its checks use assertions.

The [operator start](receipts/operator-start.json), [spawn projection](receipts/operator-spawn-public.json), [exit projection](receipts/operator-exit-public.json) and [wrapper observation](receipts/operator-wrapper-observation.json) record the direct child exit and saved-file checks. The projections omit the local PID and bind the private originals by hash; public projections cannot authenticate those originals. These are operator observations, not proof of execution or whole-process-tree cleanup. The interpreter, site machinery and dependencies were trusted.

Whole-output untouched-coordinate equality and full-vocabulary patched summaries are owner-observed; full patched logits were not retained, so their saved selected bits cannot independently be linked to a full-logit digest. No pretrained experiment was rerun during independent review.
