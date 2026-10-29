# DistilGPT2 trace correctness smoke: CPU v1

These records cover one fixed short input under `docs/PRETRAINED_SMOKE_V1.md`.
The recorded code revision is `170c7e5ab6caa995074701b20bb6affca29c3480`;
all three runs used that same detached source checkout and the same machine.
The checkpoint is `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, loaded offline after verification of
its seven required assets. Inputs are token IDs `[464, 6766, 318, 4171, 13]`,
with seed 0, float32 and one PyTorch intra-op thread. See the machine-readable
records for the complete fixed configuration and recorded environment.

`reference/result.json` reports that baseline, independent hooks, direct taps,
gradient reporting, and ID-only tracing met the protocol checks. Each mode
reported 18 events; direct and gradient each retained 110,592 tensor bytes and
ID retained none. Both `replay/results.json` and
`independent-replay/results.json` report `reproduced` with no differences.
They are fresh-process replays on the same host, not independent hardware or
statistical replicates. Two reviewers separately checked the reference and
first replay's source blobs, model assets, event metadata, value-hash
relationships and saved file digests. One of them ran the third replay.

SHA-256 of the three terminal records:

- `reference/result.json`: `7ef29dc5cb86f77fd55c1511da9af4290b819f97ec6017a9078a2ca3f416bd17`
- `replay/results.json`: `c60d017a438c469f42bec571477be239c59f837a0837f05210c05b7f323dce2e`
- `independent-replay/results.json`: `0c885a7ec1325f1976d77fd214944f3cddc5dd05711ab5173a6a334341444192`

The operator launched the reference and first replay as separate Python CLI
processes using the same empty external bytecode-cache prefix, with bytecode
writes disabled and the prefix checked empty before, between and after the
runs. The independent reviewer used a newly created separate empty prefix and
checked it before and after. CUDA was hidden via an empty
`CUDA_VISIBLE_DEVICES`; offline mode was enabled. These are operator
observations, not proof of executed bytecode or complete startup provenance.
The checkout root had no unexpected root-level Python import candidates when
inspected after the reference and before replay. The source record explicitly
states `bytecode_attested: false`.

Exact saved hashes do not provide tolerance-based comparisons across devices or
reconstruct the underlying tensors. This is a correctness smoke for the fixed
input, not a performance benchmark, a causal intervention study, or evidence
of a model's reasoning. The earlier exploratory heredoc run is separate and
has no saved v1 result.
