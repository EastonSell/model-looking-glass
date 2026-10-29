# Fixed tokenizer-only antonym cases

The [55 planned inputs](../experiments/bert_antonym_preference_public_cases_v1.json)
implement the independently reviewed [antonym protocol](DISTILBERT_ANTONYM_PREFERENCE_V1.md),
revision `4104b5ac8eb83904839fc60f7b16489a228eb481`. They contain 12 screen
rows, 16 same-pair held-outs in total, and the localities specified in the
protocol. Whichever candidate is selected has four held-outs and 12 localities
for evaluation. This is a planned reservation; no antonym model forward or
unknown/private/pretraining exposure claim follows.

The manifest SHA-256 is
`284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386`.
An independent tokenizer-only audit checked all 55 texts, masks and roles and
126 complete-sentence single-token fills against the pinned tokenizer. It
found no within-manifest complete/SEP-stripped prefix relation. The
[planned exposure record](../experiments/bert_antonym_preference_public_exposures_v1.json)
marks all 55 inputs planned.

The [named-public comparison](../experiments/bert_antonym_preference_public_comparison_v1.json)
uses main `8b61b7513f3c4a3ff9227ef5310b4a5692c37d90`, 30 public case/exposure
files, 1,393 row references and 76,615 candidate/reference pairs. It found no
exact or either-way prefix relation with either terminal SEP present or
removed. A separate independent replay reproduced the same source identities
and counts. This point-in-time public check does not certify histories outside
those named files. A separate 160-row multisuffix proposal, then tokenizer-only
and not forwarded, was compared with all 55 rows: 8,800 pairs and no such
relation. Its SHA-256 was
`bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e`;
it is a separately identified pending proposal, not part of the public-tree
comparison above. Recheck current public and coordinated pending inputs before
any model forward.

The [first tokenizer draft](../experiments/bert_antonym_preference_first_draft_v1.json)
and [first comparison](../experiments/bert_antonym_preference_first_comparison_v1.json)
are retained as superseded evidence. Before any antonym forward, that check
found the violin locality already planned in the prior color inventory; the
protocol was publicly revised to the trumpet locality. The corrected manifest
changes only that row and its protocol pins from the first draft; the 12 screen
inputs remain unchanged.

Rebuild the manifest, comparison and planned exposure into new files using the
pinned local tokenizer and the published builders. Each command uses exclusive
creation; replace `SNAPSHOT` with the locally verified pinned snapshot path.
No model or GPU is needed.

```sh
python -c "from pathlib import Path; Path('work').mkdir(exist_ok=True)"
python scripts/build_distilbert_antonym_cases_v1.py --snapshot SNAPSHOT --protocol docs/DISTILBERT_ANTONYM_PREFERENCE_V1.md --output work/antonym-cases-rebuilt.json
python scripts/check_distilbert_antonym_public_v1.py --repo . --commit 8b61b7513f3c4a3ff9227ef5310b4a5692c37d90 --cases work/antonym-cases-rebuilt.json --output work/antonym-comparison-rebuilt.json
python scripts/build_distilbert_antonym_exposures_v1.py --cases work/antonym-cases-rebuilt.json --comparison work/antonym-comparison-rebuilt.json --output work/antonym-exposure-rebuilt.json
```

The tokenizer builder checks the pinned tokenizer JSON/vocabulary and full
sentence replacement at the mask. The comparison reads the exact Git objects
at the stated commit; it may require those objects to be fetched first. The
case, comparison and exposure bytes are independently SHA-256 bound. The
separately reviewed screen owner and currentness check remain prerequisites
for the planned baseline.
