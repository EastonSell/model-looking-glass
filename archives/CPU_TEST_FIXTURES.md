# Retained CPU test evidence fixtures

Fifteen original, byte-identical records were restored to the active tree after CPU testing identified real fixture dependencies. They total 46,799,138 bytes; keeping them avoids network-dependent tests, blanket skips, or fabricated replacements. These are prior observations, not new model runs. Their original Git blobs at `3cbdec78134855532811a58a64e8dd5d4d457185` are unchanged.

| Retained path | Bytes | Original Git blob |
|---|---:|---|
| [results/bert-agreement-cpu-evaluation-v1-20261029/evidence/events.jsonl.gz](../results/bert-agreement-cpu-evaluation-v1-20261029/evidence/events.jsonl.gz) | 3,642,930 | `fc6a4344223d80120000cdebade34134941b796c` |
| [results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz](../results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz) | 455,929 | `fa3f4781c9b8ef7da8d2f8979cb8ff8dbb5323a3` |
| [results/bert-matched-dose-v1-20261029/events.jsonl.gz](../results/bert-matched-dose-v1-20261029/events.jsonl.gz) | 2,384,683 | `4eebcd7664cfbc684579d1acdfeb8adf262851b2` |
| [results/targeted-edit-cpu-v1-20261028/reference/original.f32](../results/targeted-edit-cpu-v1-20261028/reference/original.f32) | 2,613,364 | `2e0217f9a04fd6e9db1d96434c478cc130a3c4dc` |
| [results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32](../results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32) | 3,216,448 | `8a04d5defa30071f0ea2b3e094e0956537143f91` |
| [results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32](../results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32) | 1,115,136 | `6e819e1767fbf8294e99d9387bce368e34b6b5ee` |
| [results/targeted-edit-cpu-v1-20261028/reference/gradient.f32](../results/targeted-edit-cpu-v1-20261028/reference/gradient.f32) | 2,613,364 | `3c0caeadc3c8f2232d6e9c2df9f189a899c63733` |
| [results/targeted-edit-cpu-v1-20261028/reference/random.f32](../results/targeted-edit-cpu-v1-20261028/reference/random.f32) | 2,613,364 | `2ee23408aea389c8da61e2ca944201a9560a96ef` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/matched.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/matched.f32) | 4,020,560 | `3265dd1361141f03dc50ee2ae89fc93ad0a7e876` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/nuisance.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/nuisance.f32) | 4,020,560 | `631c1020c250b766149ce056d3a2fc5e752a33d9` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/opposite.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/opposite.f32) | 4,020,560 | `8f5221c784bca53494cddf17e7c0f9189fd77269` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/original.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/original.f32) | 4,020,560 | `c0a046c7bb43468b58fac82311e1b3f7ca5205f2` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/projected.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/projected.f32) | 4,020,560 | `070c3ad86deb740c4c641fe91072f7ebc35d010d` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/projected_random.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/projected_random.f32) | 4,020,560 | `bc94f6f43b7789fd7aeb0aa32ac0250d1f5d373d` |
| [results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/unprojected.f32](../results/targeted-edit-screened-matched-v1-20261029/evaluation/evaluation/unprojected.f32) | 4,020,560 | `7cdc19b68e7c4a3c881f8ecd077ecf550bd6e71a` |

The compressed BERT journals support saved-record parser, profile and evidence tests. The DistilGPT2 logits supply the public binding/mutation-guard fixture. The copying attention tensor supports recomputation of the published discovery head ranking and its follow-up binding. The targeted-edit float32 references supply original, gradient-edit and random-edit logits to the synthetic handoff and numerical-normalization checks. Their return does not imply that the historical model or pinned CUDA/Python environment was rerun. Tests requiring that exact runtime remain explicitly constrained.

The remaining archive has 2,099 files / 2,783,713,274 bytes. Its manifest excludes these retained fixtures.

The seven screened-evaluation matrices retain all original conditions, not just the positive comparison. The post-hoc locality anchor verifies all seven matrix hashes and all 140 row hashes before analysis. Restoring these bytes is not a new model run and does not alter the original locality failure.
