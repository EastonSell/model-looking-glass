# DistilBERT taste-context patching: public CPU exploration

For four fixed public sentence frames, transferring the contextual ingredient-token vector between sugar and lemon changed the masked model’s `sweet` minus `sour` logit margin toward the donor in both directions at each of blocks 0–4. Each of these 40 signed effects had the donor-directed sign. Patches at block 5 and all ingredient self-patches had zero effect on the mask logits. These are observations for one checkpoint and four convenience frames, not evidence of a general taste concept or held-out performance.

The unchanged-token (`the`) transfer is a location comparison: bidirectional context may already carry ingredient information there. Its effects are small here, sometimes opposite to the ingredient transfer, and are not assumed to be zero. The final-block zero is an architecture sanity check because the remaining masked-LM head is position-wise.

## Clean baselines

The score is a two-token logit difference, equivalent to their log odds, not the probability of a correct cloze completion. In particular, two sugar frames favor `sour` over `sweet`; in several frames other vocabulary items dominate. All probabilities use the full vocabulary.

| Frame | Ingredient | Sweet − sour | P(sweet) | P(sour) | Top five tokens |
| --- | --- | ---: | ---: | ---: | --- |
| tea | sugar | +0.635176 | 0.0787 | 0.0417 | `sweet`, `good`, `bitter`, `better`, `delicious` |
| tea | lemon | -0.239750 | 0.0473 | 0.0602 | `bitter`, `sour`, `good`, `delicious`, `lemon` |
| drink | sugar | -0.981817 | 0.0421 | 0.1125 | `bitter`, `sour`, `sweet`, `good`, `delicious` |
| drink | lemon | -1.392542 | 0.0293 | 0.1178 | `bitter`, `sour`, `good`, `better`, `delicious` |
| sauce | sugar | -0.329106 | 0.0270 | 0.0376 | `differently`, `strongly`, `sour`, `bitter`, `different` |
| sauce | lemon | -0.886889 | 0.0082 | 0.0200 | `differently`, `strongly`, `directly`, `different`, `faintly` |
| water | sugar | +0.847921 | 0.0904 | 0.0387 | `sweet`, `salty`, `bitter`, `good`, `sour` |
| water | lemon | -0.139110 | 0.0544 | 0.0625 | `salty`, `sour`, `bitter`, `sweet`, `pleasantly` |

## Per-frame intervention effects

Values are patched recipient margin minus that recipient’s clean margin. `Sugar → lemon` patches the lemon sentence with the sugar donor; `lemon → sugar` is the reverse. The last two columns apply the same donor directions at the unchanged `the` position. These rounded values include every frame and layer; exact float32-derived values and all 144 effects, including self-patches, are in [result.json](run/result.json).

| Frame | Block | Ingredient sugar → lemon | Ingredient lemon → sugar | `the` sugar → lemon | `the` lemon → sugar |
| --- | ---: | ---: | ---: | ---: | ---: |
| tea | 0 | +0.779268 | -0.795372 | +0.003119 | -0.003210 |
| tea | 1 | +0.777092 | -0.748418 | +0.000937 | +0.000121 |
| tea | 2 | +0.719883 | -0.658075 | -0.023556 | +0.021705 |
| tea | 3 | +0.666636 | -0.661144 | -0.024529 | +0.023683 |
| tea | 4 | +0.384820 | -0.411441 | -0.003636 | +0.000893 |
| tea | 5 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| drink | 0 | +0.339453 | -0.307442 | -0.007637 | +0.009205 |
| drink | 1 | +0.475281 | -0.433776 | -0.011880 | +0.017201 |
| drink | 2 | +0.343518 | -0.358375 | -0.032207 | +0.033898 |
| drink | 3 | +0.612475 | -0.683284 | -0.033728 | +0.033654 |
| drink | 4 | +0.463603 | -0.546552 | -0.032985 | +0.021615 |
| drink | 5 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| sauce | 0 | +0.781109 | -0.766253 | -0.004374 | +0.005280 |
| sauce | 1 | +0.779451 | -0.768340 | -0.010915 | +0.014272 |
| sauce | 2 | +0.537832 | -0.561937 | -0.028590 | +0.026080 |
| sauce | 3 | +0.418185 | -0.452592 | -0.039944 | +0.045663 |
| sauce | 4 | +0.218859 | -0.230448 | +0.002386 | -0.003018 |
| sauce | 5 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| water | 0 | +0.875309 | -0.894279 | +0.000414 | +0.001136 |
| water | 1 | +0.876356 | -0.856797 | -0.007536 | +0.007709 |
| water | 2 | +0.797319 | -0.767755 | -0.011435 | +0.016520 |
| water | 3 | +0.540681 | -0.598748 | -0.011484 | +0.010322 |
| water | 4 | +0.173615 | -0.211808 | +0.015346 | -0.018126 |
| water | 5 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

Every ingredient self-patch and both fresh final clean forwards per frame reproduced the full mask-logit bits. The child also checked bitwise selected-vector landing, other coordinates in the patched output, and before/after model-state equality. The recorded selected pre/post/donor bits permit independent landing checks; the full block tensors were not retained, so untouched-coordinate equality is an in-run observation. Patched full-vocabulary probabilities and top-five predictions are also in-run observations.

## Evidence and scope

The [protocol](../../docs/BERT_TASTE_CONTEXT_V1.md) and [eight exact public input rows](../../experiments/bert_taste_public_cases_v1.json) were published at source commit `bb2ccb0d4b674a4e2c90be4952ccff5195e8730a` before the run in this session. The [exposure successor](../../experiments/bert_taste_public_exposures_v2.json) records that all eight reserved row identifiers have completed-forward records. The journals do not retain the model-entry token IDs independently; the published cases and owner code bind the identifiers to the intended inputs. The offline loader checked the pinned `distilbert/distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be`; its weights SHA-256 is `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`. The result records the remaining asset hashes, CPU runtime and numeric settings. All eight rows were compared with two cited public exposure inventories; unknown prior and pretraining exposure cannot be excluded.

The [result](run/result.json), [observer receipt](receipts/observer.json), [no-model preflight](receipts/preflight-observer.json), [start record](run/started.json), and complete per-frame journals ([tea](run/tea.jsonl), [drink](run/drink.jsonl), [sauce](run/sauce.jsonl), [water](run/water.jsonl)) preserve the run. The parent recorded a zero exit and reaped its direct child, an empty private bytecode prefix, unchanged selected source bytes, and matching journal/result hashes. A separate post-exit process query found no matching taste Python process; neither is a whole-process-tree proof. The Windows virtual-environment executable can redirect to a descendant. Source hashes, local receipts and reported ordering do not authenticate execution, dependency bytes or publication chronology.

Two independent no-model readers reconstructed 248 starts/completions, all 144 effects, the eight saved full baseline rows, selected donor landing, control records, summaries and recorded hashes: [registry report](audit/registry_report.json) and [reproducibility report](audit/repro_report.json). Their [first reader](audit/registry_audit.py), [second reader](audit/bert_taste_independent_audit.py) and [fabricated tests](audit/test_bert_taste_independent_audit.py) are included. The latter reader uses integer float32 decoding and was tested under denormal flush on and off. These are saved-record consistency checks, not independent model reruns.
