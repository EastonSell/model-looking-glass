# Shared-load CPU smoke, 2026-10-28

This is a functionality smoke check of the version-1 throughput harness, not a
performance study. It used the pinned DistilGPT2 revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa` on CPU, one model and one broker,
with two public short prompts per team, one warmup of each job per mode, and a
single measured window per mode. The exact source was clean Git commit
`a36c83c3adecca7740a24ce2d02902d72e08b211`; the run records source and model
asset SHA-256 hashes.

Both modes completed four measured requests, with no rejections, failures,
cancellations, or invalid outputs. All four matched requests had bit-exact
finite logits across modes. The broker stopped cleanly and the run's final
status is `complete`. The raw rows, manifests, summary, file hashes and cleanup
record are preserved in
[`results/throughput-cpu-smoke-v1-20261028`](../results/throughput-cpu-smoke-v1-20261028/).

The windows contain only four observations each, and the host was not reserved
for an isolated latency measurement. The recorded rates and percentiles exist
for replay and accounting checks; they must not be treated as a speedup,
confidence interval, GPU benchmark, or general claim about transformers. These
forwards did not capture traces or test causal explanations. The versioned
[protocol](THROUGHPUT_PROTOCOL_V1.md) specifies the measurement boundaries and
limitations. CUDA execution remains gated pending supervised physical-device
ownership.
