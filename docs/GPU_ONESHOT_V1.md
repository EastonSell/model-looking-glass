# Supervised DistilGPT2 GPU memory pilot, v1

This protocol records a single exploratory observation of a pinned pretrained
DistilGPT2 model on one GPU. It is not a latency or throughput benchmark, a
trace experiment, a service capacity profile, or evidence of causal reasoning.
The input is the fixed two-token sequence `[15496, 995]`; seed 17, float32,
eager attention, evaluation/inference mode, and no KV cache are used. The
checkpoint and tokenizer files are size- and SHA-256-verified offline.

The parent records its source commit and verifies the exact bytes of the
runner, checkpoint loader, device identity, device lease and supervisor against
that commit. Start the controller with `-B -X pycache_prefix=<fresh-empty-dir>`.
The child starts separately with a replaced environment, a fresh bytecode
prefix and the physical GPU UUID, takes the shared device lease and verifies
the actual CUDA runtime UUID before model allocation. It refuses less than
3 GiB device-wide free memory and sets a 25% PyTorch allocator limit. That
limit does not bound CUDA context, native-library or other-process memory.

The child synchronizes and records device-wide free/total bytes and current
and peak PyTorch allocated/reserved bytes before copying the model, after the
copy, after the forward and after disposing local model/input/output aliases.
It verifies model parameters, input and output actually reside on `cuda:0`,
the output is finite and has shape `[1, 2, 50257]`, and records a SHA-256 of
the output's raw float32 bits and byte order, not the output values. The hash
identifies this observation; it does not establish cross-device bit equality.
The child synchronizes, empties the allocator cache, requires zero current
allocated/reserved bytes, attests clean and retains its lease until exit. The
parent publishes the candidate only after the supervisor reaps the matching
child and validates its clean attestation. A failed or unknown teardown leaves
the supervisor's unsafe marker; do not automatically remove it.

The parent also records `nvidia-smi` memory snapshots before and after the
child and queries whether the reaped PID is listed. Device-wide numbers can
change with desktop and other processes, and Windows WDDM may hide processes
or per-process memory. The cooperative lock cannot stop nonparticipants.
CUDA memory totals reported by device properties and `mem_get_info` are
recorded independently. No isolated timing window is claimed. A public result
omits the physical UUID and local paths; bounded private failure diagnostics
remain in the temporary run directory.

CPU-only fake checks: `python -m unittest discover -s tests -p test_gpu_oneshot.py -v`.
For a real run supply the common workspace root, the verified offline cache,
and a new absolute result path to `python -B -X pycache_prefix=<empty-dir>
-m looking_glass.gpu_oneshot --workspace-root <root> --cache-dir <cache>
--output <result>`. Coordinate a non-overlapping lease window first.
