# DistilBERT color preference v2: baseline screen

The fixed nine-row screen was **baseline-ineligible**: none of banana, strawberry or lemon passed all three required rows. Four of the nine individual factual-minus-purple margins exceeded the fixed threshold. The recorded attempt stopped at the screen; it performed no parameter edit, held-out evaluation or locality evaluation. This is a baseline result, not a negative edit result or a conclusion that the model lacks color knowledge.

This outcome-informed follow-up was fixed after the [earlier full-top screen](../distilbert-color-edit-top-v1-20261029/screen/README.md) had stopped ineligible. The [v2 protocol](../../docs/DISTILBERT_COLOR_PREFERENCE_V2.md) and [37-case manifest](../../experiments/bert_color_preference_public_cases_v2.json) were public before this forward. For each candidate, the screen required each of its discovery and two calibration rows to have factual-minus-purple logit margin strictly greater than decimal 0.10. The decision compares the exact saved binary32 values against exact 1/10; rounded values below are descriptive. This tests one masked-token preference, not a generated answer or top-token factual edit.

| Object | Screen row | Factual | Margin | Pass | Full-vocabulary top | Factual rank | P(factual) | P(purple) | Pair mass |
| --- | --- | --- | ---: | :---: | --- | ---: | ---: | ---: | ---: |
| Banana | discovery | yellow | +0.315461 | yes | reddish | 2 | 0.076782 | 0.056009 | 0.132791 |
| Banana | calibration 01 | yellow | +0.511633 | yes | yellow | 1 | 0.130672 | 0.078340 | 0.209013 |
| Banana | calibration 02 | yellow | -0.637424 | no | purple | 4 | 0.063969 | 0.121003 | 0.184972 |
| Strawberry | discovery | red | -1.516251 | no | reddish | 14 | 0.015616 | 0.071135 | 0.086751 |
| Strawberry | calibration 01 | red | -1.004620 | no | yellow | 10 | 0.032523 | 0.088815 | 0.121338 |
| Strawberry | calibration 02 | red | -1.469651 | no | purple | 11 | 0.028319 | 0.123122 | 0.151441 |
| Lemon | discovery | yellow | +0.146537 | yes | reddish | 2 | 0.063928 | 0.055214 | 0.119143 |
| Lemon | calibration 01 | yellow | +0.510685 | yes | yellow | 1 | 0.132096 | 0.079269 | 0.211365 |
| Lemon | calibration 02 | yellow | -0.513804 | no | purple | 4 | 0.063820 | 0.106683 | 0.170503 |

The top token is the unique full-vocabulary winner; rank is one plus the number of logits strictly greater than the factual logit. The pair's share of total vocabulary probability is shown because a preference within two tokens does not by itself show a high probability for either token. All three candidates fail the fixed all-row gate: banana and lemon fail calibration 02, and strawberry fails every row. These are nine fixed, public, template-specific observations, not independent objects or a generalization estimate.

## Saved evidence and scope

The [screen index](screen/index.json) inventories the compressed [stage archive](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilbert-color-preference-v2-20261029/screen/part-0000.gzchunk), whose SHA-256 is `a7c01d8676c349b99b09a1e2be6f721c30bd306ce730ed582e07338d36f2dd72`. Its eight members include all nine full 30,522-float mask-vocabulary logits, input/action records, result and cooperative process records; the separately indexed [run sidecar](screen/run.json) binds the receipt. The journal SHA-256 is `deab8b0abb0a5a6df3ea754b4f8c45df042b8c9f47d73417e516eaa381c1140e` and the result SHA-256 is `15e4c8d0f36387a0a162e6651db4164efca88170bb0f0021aedeabecef12b16b`.

The public owner source was `dc3914af4df43135cc701e425514046790d63f3a`; the fixed checkpoint was `distilbert/distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be`, with CPU float32/eager runtime recorded in the stage. The launch record checks 26 named public case/exposure sources at main `7673846d29f6534bedc6fa25ba591ab1acdd32cc`: 46,250 comparisons and no complete-input prefix conflict, with and without final SEP. This is a point-in-time named-public comparison, not a claim about unknown, private or pretraining exposure. The observer and direct supervisor reported successful completion; the saved process/source records are cooperative evidence, not independent execution or process-tree attestation.

Two independent [saved-score](independent/screen-review.json) and [second-score](independent/second-score-review.json) replays agree on all rows and selection; a separate [methods replay](independent/methods-review.json) also agrees. Additional saved-record checks cover the [summary](independent/summary-review.json), [receipt](independent/receipt-review.json), [22-source map](independent/source-review.json), [launch scan](independent/launch-scan-review.json), and [archive bytes](independent/bundle-review.json). Their scopes do not authenticate model execution. The receipt review uses private local command and path observations, so only its path-free report is published. A limited [path-pattern scan](independent/path-review.json) is not a complete de-identification guarantee.

The successor exposure inventory records the nine saved screen rows as observed-saved and the other 28 rows as planned. Planned means the completed recorded attempt did not forward those rows, with no claim about unknown prior or later use.

## Model-free replay

From the repository root, the following commands extract the saved journal without overwriting an existing file and recompute the fixed selector, full-vocabulary ranks, probabilities and pair mass. The reader and its two adjacent helpers use only the Python standard library; they require normal binary32/binary64 subnormal handling and do not load a model. The archive and manifest hashes above should be checked before relying on the result.

```sh
python -c "import hashlib,tarfile; from pathlib import Path; p=Path('results/distilbert-color-preference-v2-20261029/screen/part-0000.gzchunk'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='a7c01d8676c349b99b09a1e2be6f721c30bd306ce730ed582e07338d36f2dd72'; t=tarfile.open(p,'r:gz'); b=t.extractfile('events.jsonl').read(); t.close(); assert hashlib.sha256(b).hexdigest()=='deab8b0abb0a5a6df3ea754b4f8c45df042b8c9f47d73417e516eaa381c1140e'; Path('work').mkdir(exist_ok=True); f=Path('work/color-preference-v2-events.jsonl').open('xb'); assert f.write(b)==len(b); f.close()"
python results/distilbert-color-preference-v2-20261029/independent/review_distilbert_color_preference_screen_v2.py --manifest experiments/bert_color_preference_public_cases_v2.json --journal work/color-preference-v2-events.jsonl --output work/color-preference-v2-independent-replay.json
```

The reference saved-logit replay report has SHA-256 `657f41521515edfbd2bb8aa05df25252adf9f3b89426ff35e484982cef3a5241`; floating-point serialization may vary across supported systems. No model, GPU or new study input is needed for replay.
