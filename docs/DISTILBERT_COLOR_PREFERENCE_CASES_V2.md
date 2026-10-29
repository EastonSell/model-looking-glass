# Masked-color preference cases: tokenizer-only reservation

The [outcome-informed protocol](DISTILBERT_COLOR_PREFERENCE_V2.md) was fixed
and independently reviewed before this case inventory was frozen. It tests a
two-color masked-token preference, not a generated answer or a factual
full-vocabulary top. These cases are public, and the held-out target phrasings
refer to the same selected object as discovery. No v2 model forward was run
to construct this inventory.

The corrected protocol is merge `d94594f69426aa5e41a7cf90acd1e98be42f9087`
and has SHA-256 `41ffb6f85d0017a3644b4ee5cc7328ffa1c90c3fec36faeb438fe96625dfc34d`.
The initially published green-only lime answer was corrected before any v2
forward to accept both green and yellow. The first tokenizer-only draft is
retained at [pre-forward-draft.json](../results/distilbert-color-preference-v2/pre-forward-draft.json),
SHA-256 `b6f3e0d9f417f992abfa69361d70437afa201afec3947a932290ba428e0423a0`.
Its 37 input texts and token rows are identical to the corrected inventory;
the lime accepted set and protocol identity changed.

The corrected [case inventory](../experiments/bert_color_preference_public_cases_v2.json)
has SHA-256 `f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10`.
It has 37 unique input rows and 80 valid full-context single-token fills.
Two independent reviewers reproduced the protocol text, tokenizer output,
mask positions and fill boundaries without loading a model. The
[exposure record](../experiments/bert_color_preference_public_exposures_v1.json)
marks every row **planned**; it records no execution or authorization to run.

The [public comparison](../experiments/bert_color_preference_public_comparison_v2.json)
has SHA-256 `6c29d8875238448a8fc20887832b02d058b6f76fcab0a41fe6385050e932a1c3`.
It compares the 37 planned rows with 577 row references from 14 named public
snapshots, including the original color study, the cue-dose observed successor
and the fixed cross-model inputs. There are 21,349 cross-source comparisons
and 666 within-inventory pairs, with zero equal or either-way complete-prefix
relations before or after removing one terminal `[SEP]`. An independent
replay reached the same counts. These are named-snapshot checks only; repeated
rows are counted as separate references. They do not establish non-overlap
with unknown or private data, currentness after the cited snapshots, or
absence from pretraining.

To reproduce with the pinned local tokenizer snapshot and `tokenizers==0.22.2`,
use a new output path. The builder verifies the protocol, tokenizer and vocab
hashes and does not load a model:

```text
python -B scripts/build_distilbert_color_preference_cases_v2.py --snapshot PATH_TO_PINNED_SNAPSHOT --protocol docs/DISTILBERT_COLOR_PREFERENCE_V2.md --output work/color-preference-rebuilt.json
python -B scripts/check_distilbert_color_preference_exposures_v2.py --repository . --cases experiments/bert_color_preference_public_cases_v2.json --output work/color-preference-comparison-rebuilt.json
python -B scripts/build_distilbert_color_preference_exposures_v1.py --cases experiments/bert_color_preference_public_cases_v2.json --comparison experiments/bert_color_preference_public_comparison_v2.json --output work/color-preference-exposures-rebuilt.json
```

Each output is created exclusively; create `work/` first if needed. A later
executable, fresh relevant-history check, independent review and retained
stage observations are still required before any proposed screen.
