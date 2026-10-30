# Retained CPU test evidence fixtures

Six original, byte-identical records were restored to the active tree after CPU testing identified real fixture dependencies. They total 13,428,490 bytes; keeping them avoids network-dependent tests, blanket skips, or fabricated replacements. These are prior observations, not new model runs. Their original Git blobs at `3cbdec78134855532811a58a64e8dd5d4d457185` are unchanged.

| Retained path | Bytes | Original Git blob |
|---|---:|---|
| [results/bert-agreement-cpu-evaluation-v1-20261029/evidence/events.jsonl.gz](../results/bert-agreement-cpu-evaluation-v1-20261029/evidence/events.jsonl.gz) | 3,642,930 | `fc6a4344223d80120000cdebade34134941b796c` |
| [results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz](../results/bert-agreement-dev-v1-20261028/attempt-01/events.jsonl.gz) | 455,929 | `fa3f4781c9b8ef7da8d2f8979cb8ff8dbb5323a3` |
| [results/bert-matched-dose-v1-20261029/events.jsonl.gz](../results/bert-matched-dose-v1-20261029/events.jsonl.gz) | 2,384,683 | `4eebcd7664cfbc684579d1acdfeb8adf262851b2` |
| [results/targeted-edit-cpu-v1-20261028/reference/original.f32](../results/targeted-edit-cpu-v1-20261028/reference/original.f32) | 2,613,364 | `2e0217f9a04fd6e9db1d96434c478cc130a3c4dc` |
| [results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32](../results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32) | 3,216,448 | `8a04d5defa30071f0ea2b3e094e0956537143f91` |
| [results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32](../results/distilgpt2-copying-dev-v1-20261028/run/attentions.f32) | 1,115,136 | `6e819e1767fbf8294e99d9387bce368e34b6b5ee` |

The compressed BERT journals support saved-record parser, profile and evidence tests. The DistilGPT2 logits supply the public binding/mutation-guard fixture. The copying attention tensor supports recomputation of the published discovery head ranking and its follow-up binding. The targeted-edit float32 reference supplies an original logit row to the synthetic targeted-edit handoff integration. Their return does not imply that the historical model or pinned CUDA/Python environment was rerun. Tests requiring that exact runtime remain explicitly constrained.

The remaining archive has 2,108 files / 2,817,083,922 bytes. Its manifest excludes these retained fixtures.
