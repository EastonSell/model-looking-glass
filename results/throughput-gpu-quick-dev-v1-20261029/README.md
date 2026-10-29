# Shared-model GPU throughput: quick development run

This is one small, unprofiled development run on an RTX 2070 SUPER. It
demonstrates that two clients can submit to a single resident DistilGPT2 owner;
it does not establish a stable throughput difference, isolated latency,
scheduler fairness, or a safe admission profile. The controller's authoritative
status is `development_only`; its nonzero exit is intentional for an
unprofiled run, while the supervised owner exited zero and was reaped cleanly.

The fixed workload consisted of four batch-one requests per window: two public
five-position prompts for each of two teams, with seeds 1000 and 1001. A
serial closed-loop client and two concurrent closed-loop team clients submitted
the same frozen jobs, with at most one outstanding request per client. One
broker owner executed forwards serially; there was no batching, activation
capture, or claim of overlapping GPU kernels. Each mode had four successful
warmups before its measured window. The seeded order was serial, then shared.

| Measure | Serial client | Two concurrent clients |
| --- | ---: | ---: |
| Successful / attempted requests | 4 / 4 | 4 / 4 |
| Window duration | 89.7493 ms | 96.0269 ms |
| Successful requests per second | 44.57 | 41.65 |
| Input positions per second | 222.84 | 208.27 |
| Ready-to-dispatch queue wait, p50 / p95 | 0.075 / 0.079 ms | 18.270 / 30.307 ms |
| Worker service, p50 / p95 | 19.940 / 21.757 ms | 22.066 / 29.783 ms |
| Client response, p50 / p95 | 22.227 / 24.008 ms | 40.696 / 56.166 ms |

The observed shared/serial throughput ratio was about 0.935 in this pair. Four
requests per window are too few to infer a general performance change; the
loads have different arrival patterns, and the serial client submits team A's
jobs before team B's. The queue and response timings exclude time before each
client attempts a request. With four observations the nearest-rank p95 is the
maximum, not a reliable tail estimate. Each team completed two requests in
each window; this does not establish fairness. There were no failed, cancelled,
rejected, unresolved, invalid-output, or submission-error attempts. The four
corresponding serial/shared result digests matched. The original logits were
not retained, so the saved digests cannot independently establish their
finiteness or reproduce their values without another execution.

Per-window PyTorch allocator peaks were 369,520,128 bytes allocated
(approximately 352.4 MiB) and 406,847,488 bytes reserved (388 MiB). Both
device-free snapshots were 7,018,119,168 of 8,589,606,912 bytes. The PyTorch
per-process caching-allocator cap was 25%; it does not cap CUDA context,
native, or other-process memory. The 2 GiB model-load and 1 GiB per-forward
ledger reservations are provisional estimates, not measured bounds or proof
that any other workload fits. WDDM and other GPU activity were not isolated.
After the broker drained, synchronized cleanup recorded 32 MiB before the
version-bound private cuBLAS workspace clear and zero allocated and reserved
bytes afterward; the owner attested clean and the supervisor reaped it.

Windows use `perf_counter_ns`/QPC, with a recorded nominal resolution of
100 ns. GPU work is synchronized at the worker boundaries. Rates use the full
barrier-release-to-terminal wall window and include the prescribed output
handling and CPU validation/hash work. They are not isolated model-forward
latencies.

The model was `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, in float32 eval mode with eager
attention and no cache. The source and protocol are pinned to commit
`c480e7e987ef7127ae61658ead3018e15ce9ae84` and
[`THROUGHPUT_PROTOCOL_V1.md`](../../docs/THROUGHPUT_PROTOCOL_V1.md). Source
and asset digests, seeds, settings, and workload identities are recorded in
the evidence. These hashes are consistency checks, not cryptographic proof of
execution; the runtime dependencies and checkout are trusted not to change
during the run.

## Evidence and redaction

The original private run is preserved locally. This directory contains a
separately hashed public derivative: physical GPU UUID strings were replaced
with `<GPU_UUID_REDACTED>` and process IDs with `null`; all other bytes were
preserved. [redaction_manifest.json](redaction_manifest.json) records the
original and public SHA-256 hashes and the number of replacements per file.
The three event streams under `evidence/owner/measurements/`—
`warmup_samples.jsonl`, `raw_samples.jsonl`, and `partial_windows.jsonl`—are
byte-for-byte identical to the originals. Relative event intervals, outcome
counts, output digests, and memory data can be recomputed from the public
records. Original evidence files contain hashes of original bytes, so those
embedded references intentionally do not match files altered by redaction;
the publication inventory distinguishes the two. No signature or independent
attestation of the original execution is implied.

Two independent CPU-only reviews recomputed the raw counts, timings, rates,
output-digest matches, source and evidence hashes, and clean teardown. The
cleanup patch was independently reviewed and its focused integrated CPU suite
passed 223 tests.
