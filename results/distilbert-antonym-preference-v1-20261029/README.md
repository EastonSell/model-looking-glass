# DistilBERT masked-antonym preference: baseline screen

The fixed 12-row screen was **baseline-ineligible**: none of the four candidates met the discovery unique-top rule, so no candidate was selected. The recorded attempt stopped at the screen; no parameter edit, held-out evaluation or locality test was run. This is a baseline result for these prompts, not a failed edit or a conclusion that the model lacks antonym knowledge.

The [protocol](../../docs/DISTILBERT_ANTONYM_PREFERENCE_V1.md) and [55-case inventory](../../experiments/bert_antonym_preference_public_cases_v1.json) were public before model entry, following the earlier [color-preference screen](../distilbert-color-preference-v2-20261029/README.md). A candidate required the preferred factual token to be the strict unique full-vocabulary winner on discovery, and factual-minus-cue logit margin strictly greater than exact rational 1/10 on discovery and both calibration rows. All 12 rows were run before the fixed candidate-order selection. The margin is computed from the exact saved binary32 logits; rounded displays do not decide the gate.

| Object | Row | Factual / cue | Margin | Row pass | Top | Ranks factual / cue / red | P(factual) | P(cue) | P(red) | Pair mass |
| --- | --- | --- | ---: | :---: | --- | --- | ---: | ---: | ---: | ---: |
| box | discovery | small / large | +0.959358 | no | zero | 61 / 153 / 1624 | 0.002956 | 0.001132 | 0.000031 | 0.004088 |
| box | calibration-01 | small / large | +1.444216 | yes | rectangular | 2 / 15 / 333 | 0.025081 | 0.005917 | 0.000467 | 0.030999 |
| box | calibration-02 | small / large | +2.387647 | yes | rectangular | 3 / 5 / 888 | 0.108404 | 0.009956 | 0.000092 | 0.118361 |
| rope | discovery | short / long | +0.151824 | no | zero | 262 / 292 / 1664 | 0.000529 | 0.000454 | 0.000027 | 0.000983 |
| rope | calibration-01 | short / long | +1.805346 | yes | short | 1 / 6 / 831 | 0.148418 | 0.024402 | 0.000093 | 0.172820 |
| rope | calibration-02 | short / long | +2.694641 | yes | short | 1 / 6 / 1016 | 0.311023 | 0.021015 | 0.000026 | 0.332037 |
| drink | discovery | cold / hot | -0.605340 | no | zero | 269 / 168 / 670 | 0.000482 | 0.000883 | 0.000124 | 0.001366 |
| drink | calibration-01 | cold / hot | -1.025102 | no | hot | 2 / 1 / 111 | 0.118750 | 0.331002 | 0.000598 | 0.449753 |
| drink | calibration-02 | cold / hot | +0.521831 | yes | soft | 2 / 3 / 157 | 0.210870 | 0.125137 | 0.000248 | 0.336007 |
| room | discovery | dark / bright | -1.498566 | no | zero | 279 / 74 / 399 | 0.000499 | 0.002231 | 0.000317 | 0.002730 |
| room | calibration-01 | dark / bright | -0.440250 | no | bright | 2 / 1 / 26 | 0.112316 | 0.174437 | 0.004037 | 0.286752 |
| room | calibration-02 | dark / bright | +1.056211 | yes | dark | 1 / 2 / 52 | 0.167457 | 0.058237 | 0.002314 | 0.225694 |

The top is the unique full-vocabulary winner; rank is one plus the number of logits strictly greater than that token's. The factual/cue pair mass is their combined probability under the full vocabulary. Eight of twelve margins exceed the threshold, but every discovery top is `zero`: the factual ranks are 61, 262, 269 and 279 for box, rope, drink and room, respectively. Box and rope pass all three pairwise margins but still fail the discovery-top rule. Drink and room also fail the discovery and first-calibration margins. Discovery pair mass ranges from 0.098% to 0.409%; a two-token preference can coexist with little full-vocabulary probability on either token.

The accepted set also included the following alternatives, which did not replace the preferred factual token in the discovery gate:

| Screen row | Accepted alternative | Rank | Probability |
| --- | --- | ---: | ---: |
| box-discovery | little | 1157 | 0.000053 |
| box-calibration-01 | little | 63 | 0.002094 |
| box-calibration-02 | little | 52 | 0.001662 |
| room-discovery | dim | 274 | 0.000505 |
| room-calibration-01 | dim | 21 | 0.004977 |
| room-calibration-02 | dim | 27 | 0.004569 |

These are fixed masked-token observations, not generated answers or evidence that an abstract antonym computation was changed. The four candidates share the same discovery phrasing and are not independent estimates of general capability.

## Saved evidence and scope

The [screen index](screen/index.json) inventories the compressed [stage archive](screen/part-0000.gzchunk), SHA-256 `d4d14562ef02e486299951b2b1159f79c4d6933eadbcfedc7dc29b0c46463e2b`. Its eight members retain all 12 full 30,522-float vocabulary rows, inputs, result and cooperative process records. The separately indexed [run sidecar](screen/run.json) binds the receipt. Journal SHA-256 is `7ed70ebfe6e3d05d45cf7e7f5cf6904b936faf7eefcffced7ceb63540132982e`; result SHA-256 is `ef66b2aadf515856c02ef24edd53735a20bde86f529e6e319797124d1adb5538`.

The public owner source was `cbfa224f0d2d8f1ac6f5e16942b58bb7821345e1`; the checkpoint was `distilbert/distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be`, with CPU float32 and eager execution recorded. The prepared scan checked named public inputs at main `9c2ed04085337361bb616292ff2e4ed33d6fc926`; the launch-bound scan rechecked main `1e6955f491851344f02ee83a68498914ccba815a`. The independent launch scan found 32 external source records, 1,585 row references, 87,175 row pairs and no complete-input or one-SEP-removed prefix relation. Two separately pinned public input inventories were also checked ([independent comparison](independent/additional-public-review.json)). These point-in-time named-public checks do not establish unknown, private or pretraining history.

An independent [saved-logit replay](independent/screen-review.json) and [summary comparison](independent/summary-review.json) agree with the result, as does a separate [methods replay](independent/methods-review.json). The [receipt](independent/receipt-review.json), [source and cached-asset](independent/source-review.json), [launch-scan](independent/launch-scan-review.json), and [archive](independent/bundle-review.json) checks compare saved and currently available bytes. The direct observer and supervisor reported zero exits. These cooperative observations are not independent model-execution or process-tree attestation.

The successor exposure inventory marks the 12 completed screen rows observed-saved and the other 43 fixed rows planned. Planned here means this recorded attempt did not forward them; it makes no claim about unknown prior or later use.

## Model-free replay

From the repository root, the commands below verify and extract the saved journal without overwriting an existing file, then recompute all full-vocabulary rows and the exact fixed selection. The reader uses only Python's standard library, requires normal binary32/binary64 subnormal handling, and loads no model.

```sh
python -c "import hashlib,tarfile; from pathlib import Path; p=Path('results/distilbert-antonym-preference-v1-20261029/screen/part-0000.gzchunk'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='d4d14562ef02e486299951b2b1159f79c4d6933eadbcfedc7dc29b0c46463e2b'; t=tarfile.open(p,'r:gz'); b=t.extractfile('events.jsonl').read(); t.close(); assert hashlib.sha256(b).hexdigest()=='7ed70ebfe6e3d05d45cf7e7f5cf6904b936faf7eefcffced7ceb63540132982e'; Path('work').mkdir(exist_ok=True); f=Path('work/antonym-screen-events.jsonl').open('xb'); assert f.write(b)==len(b); f.close()"
python scripts/review_distilbert_antonym_screen.py --manifest experiments/bert_antonym_preference_public_cases_v1.json --manifest-sha256 284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386 --journal work/antonym-screen-events.jsonl --output work/antonym-screen-independent-replay.json
```

The independent replay report SHA-256 is `c64862705bf087242140e88177cc5b72f4198c7a5395dd0357ef4073cb084a52`; numeric serialization can vary across supported systems.
