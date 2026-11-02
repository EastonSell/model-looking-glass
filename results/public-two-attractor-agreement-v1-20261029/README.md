# Public two-attractor agreement screen

On these 128 fixed public prompts, DistilGPT-2 favored the subject-matching member of `is`/`are` in **128/128 pairwise comparisons**. Mean subject-signed margin fell from **2.748** with both intervening nouns congruent to **1.755** with both incongruent. This is a comparison of two token logits, not the model's full-vocabulary top-token accuracy.

The prespecified interaction varies by subject number: its mean is **+0.289** for the 16 singular-subject units and **−0.073** for the 16 plural-subject units. Across all 32 units, 18 interactions are positive and 14 negative; the **+0.108** pooled mean masks this heterogeneity. The near and beside average main shifts are **−0.533** and **−0.459** logit units. These are descriptive values on this fixed design, not a test of statistical equivalence or evidence that the two effects are generally additive.

| Prespecified lexical partition | Units | Mean near shift | Mean beside shift | Mean interaction |
|---|---:|---:|---:|---:|
| Development-labeled | 8 | −0.504 | −0.181 | +0.165 |
| Evaluation | 8 | −0.592 | −0.698 | +0.078 |
| New attractor pair only | 8 | −0.515 | −0.668 | +0.073 |
| New subject only | 8 | −0.523 | −0.288 | +0.117 |

All 64 matched raw subject-number controls were positive. All 128 `was`/`were` comparisons also favored the subject-matching member. The complete per-case logits, all 32 units, each of the 64 controls, baseline counts, secondary margins and group summaries are in `result.json`; no cases were excluded. Each lexical partition contains 32 inputs. The evaluation partition was held out from local study development, but all four partitions use the same syntactic template. Prior model pretraining and unrecorded exposures remain unknown.

The protocol and cases were merged in [PR #543](https://github.com/EastonSell/model-looking-glass/pull/543) before model execution; the fixed inputs were reserved as planned in [PR #551](https://github.com/EastonSell/model-looking-glass/pull/551). The direct operator observation records a CPU process exit of 0 and a completed 128-case journal. The separately written `audit_saved.py` replayed the saved selected logits and matched every reported calculation; `independent-audit.json` is its result. That audit establishes saved-byte arithmetic consistency, not independent process-execution attestation or a full-logit replay.

The checkpoint is the pinned `distilbert/distilgpt2` revision in the protocol, with offline CPU float32/eager execution. The 128 cases used only the published input records; the original natural protected split and its consumed claim were untouched. The original 128-row public-support comparison is conditional on the earlier preparation following its published sampler and tokenizer. This one-template, four-subject, two-attractor-pair screen does not identify an internal mechanism or establish generalization to a population.

## Retained evidence

- `attempt.jsonl`: ordered start/result records, SHA-256 `9d1dabc082a075a1ef0cdc2cd7d619e7bdb1fb666ebfc4b0ba55635c7b55e2c4`.
- `result.json`: per-case selected logits and all analyses, SHA-256 `2c7a637654d00bf28f960e02b8301024573e28d4e22a8fa4cf901d2dd9ff5116`.
- `process-observation.json`: direct process and stream hashes, SHA-256 `4d100812f542e28bbb2d5e842146d65166e7d587d063b082ac5ed11a234b8aa5`; it records the observer's own report rather than an independent witness.
- `audit_saved.py`: independent saved-data reader, SHA-256 `d3d9ab5f0899aa4a60e3492d1f89a402265fe83736f0c528cfa8565916c84617`.
- `independent-audit.json`: audit result, SHA-256 `69daf89ededbbda5607ba55c32f7118f8e1cb1d7e7a2d27eeaf63d8149c103bf`.

From the repository root, replay the model-free audit with:

```text
python -B results/public-two-attractor-agreement-v1-20261029/audit_saved.py experiments/distilgpt2_two_attractor_cases_v1.json results/public-two-attractor-agreement-v1-20261029/attempt.jsonl results/public-two-attractor-agreement-v1-20261029/result.json 9d1dabc082a075a1ef0cdc2cd7d619e7bdb1fb666ebfc4b0ba55635c7b55e2c4 2c7a637654d00bf28f960e02b8301024573e28d4e22a8fa4cf901d2dd9ff5116
```
