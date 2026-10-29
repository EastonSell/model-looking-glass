# DistilGPT-2 sampled CPU batch check, v1

The four fixed public synthetic cases in the [frozen protocol](frozen.json)
all met the prespecified serial-reference tolerance in this CPU run. The
resident worker made four serial forwards and two mixed two-row forwards.
The profiler observed the intended request IDs, token and mask values, and
input shapes for each model entry. This is descriptive evidence for this fixed
sample, CPU runtime and two-row shape; it is not proof of general batch
separability, privacy, GPU correctness, shared-service throughput, or a causal
explanation.

| Case | Maximum absolute difference | Maximum relative difference |
| --- | ---: | ---: |
| case-1-1 | 3.814697265625e-5 | 6.445686614197488e-7 |
| case-1-2 | 3.0517578125e-5 | 5.244596508606547e-7 |
| case-2-1 | 3.0517578125e-5 | 4.6492311973176727e-7 |
| case-2-2 | 3.0517578125e-5 | 4.661784224655378e-7 |

The elementwise acceptance rule was `abs(mixed - serial) <= 1e-6 +
1e-5 * abs(serial)`. The maximum absolute difference exceeds the absolute
term alone; the relative term was part of the rule before the run. An
independent NumPy reconstruction of all 804,112 comparisons reproduced the
reported maxima and found a largest difference-to-tolerance ratio of about
0.06435. The ratio is descriptive for this fixed sample.

The deterministic batch-mean toy control failed all four cases, as expected.
Positive playback of the archived logits reproduced the original report.
Reversing mixed output rows in playback also failed all four cases, detecting
the deliberate cross-owner misdelivery. These controls use different bound
identities; neither is evidence that the actual model is universally safe.

The source was published at commit `58b43f3698fa8b8acf537f8a0552ec142c350622`
before the separately committed frozen manifest
`a1d1e960e9d4cb55e66ab5460d3e8f8ecc127794`, with canonical manifest SHA-256
`4f8151f6d946d4395fa49080d699a4eccb6a6416b6683810a0dd5fe6718c140d`.
The branch update preceded the run launch in this session. Independent
reviewers subsequently matched the remote manifest blob and source ancestry;
the tool history and reported clock times are not independently trusted
timestamps or proof of model execution.

The loader verified the pinned checkpoint and seven asset digests; the
checkpoint weights SHA-256 is
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The loaded model's named state digest matched before and after the forwards.
The runtime manifest records the CPU, software and numerical configuration;
the process hid GPUs and PyTorch reported CUDA uninitialized. The observer
adds overhead, so this run provides no performance measurement.

The [protocol](../../docs/DISTILGPT2_BATCH_CPU_V1.md) fixes the public EOS-token
fixtures, input and output shapes, tolerance, controls, runtime and trust
boundaries. The [primary report](run/primary-report.json),
[summary](run/summary.json), [attempt record](run/attempt.json), and
[raw logits archive](run/logits.zip) retain the result. The archive contains
6,432,896 raw little-endian float32 bytes and has SHA-256
`d1d2c114fb46b8fd810ba2c5940e62e39dc75f82ab1671349822e76bc7017562`.
The summary lists each member's request ID, shape and checksum. A separate
structural audit checked those records, manifest, source and archive; hashes
and reports do not authenticate an executor or establish that other inputs
will behave the same way.

Replaying the frozen protocol requires the committed source and manifest
version, since later changes to the worker or other listed sources are
deliberately rejected. The recorded runtime also fingerprints interpreter
paths and settings, so exact replay is tied to that environment. Relative to
the original workspace root, this run used `work/.venv/Scripts/python.exe`,
the `work/hf-cache/hub` checkpoint cache and the deliberately absent
`work/distilgpt2-batch-cpu-v1-pycache-never-created` prefix, expanded to
absolute paths at launch.

An independently launched fresh CPU process repeated the frozen protocol on
the same machine and runtime. Its six uncompressed output members match the
original member SHA-256s, and the primary report, controls, observed input
rows and model-state fingerprints match. The [replay records](replay/) retain
the attempt, report, summary and raw archive. The replay ZIP container SHA-256
is `5c7aafab931e8954d1c40792f89bbb8c2c7839db9a99224bc1be3a977b0803fb`;
its container hash differs from the original because member timestamps differ,
while the uncompressed member bytes match. This is a same-host repeat of the
four length-four synthetic cases with the same leading EOS token and all-one
masks, not an independent hardware replication. Padding and mask variants,
other lengths, natural-language prompts, GPU execution and broker batching
were not tested by this run.
