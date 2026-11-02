# DistilGPT2 16-token GPU routing observation

One supervised development run used a single resident DistilGPT2 model for
four serial forwards at shape `1 × 16`. Two threads submitted the client
requests through an in-process Broker; this does not establish that their
submissions overlapped in time. Each client’s complete 804,112 float32 logits
was bit-identical to its own same-device serial reference. The two last-active
reference rows differed on all 50,257 logits at the fixed tolerance
`1e-6 + 1e-5 * abs(reference)`, so swapping these two results would be detected
by that check.

The inputs were the all-active synthetic-16 row and the gas row with 15 active
tokens followed by one right-padding token. Both were already public, reserved
and CPU-exposed. The result is a single routing and repeatability observation,
not a held-out test, IPC or trace observation, batched forward, throughput
benchmark or general service-capacity guarantee.

The separately published [CPU reference](../short-service-cpu-reference-v1-20261029/)
contains full logits for the same padded inputs and an unpadded gas reference.
In a post-run descriptive comparison, GPU and CPU full logits were not
bit-identical: the maximum absolute difference was `9.1552734375e-5` for each
row, and zero of the 804,112 logits per row exceeded the routing tolerance
when applied to the CPU values. Last-active maximum absolute differences were
`7.62939453125e-5` for synthetic-16 and `6.103515625e-5` for gas, also with
zero violations. This cross-runtime comparison was not a prespecified pass gate.

The 21 saved synchronized memory observations reported a minimum sampled
device-wide free value of 7,016,022,016 bytes (6,691 MiB). The largest
process-local reset-window peaks were 371,566,080 bytes (354.35 MiB) allocated
and 408,944,640 bytes (390 MiB) reserved. Free memory is sampled at points;
allocator peaks are process-local and do not include every native allocation
or other process. These observations do not establish a 32-token bound.

The saved owner record says the known model/input/output references were
released and that process-local allocated/reserved counters and logical host
and device claims were zero after guarded cleanup. The supervisor and outer
controller records report clean exit 0 and a reaped direct controller; an
unlocked post-exit marker check was reported by the coordinator but was not
saved as a separate point record. None of this proves complete native CUDA
teardown, descendant absence beyond the supervised scope, global device
idleness, or independently authenticates model execution. No cancellation was
exercised.

The reviewed source was public commit
[`ca9f5a5e`](https://github.com/EastonSell/model-looking-glass/commit/ca9f5a5e6d4593cfe16c4e1cfa19800f2d6df5e7)
(tree `265395cbeefc502004e1b47e17e5983feece4898`), merged in
[#649](https://github.com/EastonSell/model-looking-glass/pull/649). The model
was `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`,
weights SHA-256 `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The runtime record reports float32 eager evaluation, disabled cache/autocast
and TF32, deterministic settings, Torch `2.8.0+cu128`, Transformers `4.57.3`
and CUDA build `12.8`. Pretraining exposure to the inputs is unknown.

The archive contains the four full-logit payloads, saved model-entry inputs,
memory and cleanup records, the result and scoped public-input checks.
`archive-index.json` pins all 12 parts and 34 members. From this directory,
reconstruct them into a new absolute destination:

```text
python -I -S -B rebuild_evidence.py --destination <new-absolute-directory>
```

The helper checks the index, each part, reconstructed archive and each member;
an interrupted extraction leaves partial files and needs a separate fresh
destination. The archive does not include the separately published CPU
reference. After reconstructing both archives and checking out the pinned
source, the model-free replay uses absolute paths:

```text
python -I -S -B replay_short_cpu_gpu_public.py --cpu-dir <cpu-extraction>/data --gpu-dir <gpu-extraction>/data --manifest <source-checkout>/experiments/short_cpu_service_reference_inputs_v1.json --source ca9f5a5e6d4593cfe16c4e1cfa19800f2d6df5e7 --bucket 16 --result-sha256 106fa110a0092fb5a6beb1884f867f2fd8e37b466613eeec50c94b5c6acd1106 --output <new-absolute-report-path>
```

The replay reads saved logits and records, with no model or GPU call. Its
published output is `short-service-gpu16-public-replay-v1.json` (SHA-256
`9ba90fadaf6986cfdd847da3215b5e11efa8de04adc144636c4848af90fe4e01`).
The signed-zero regression makes temporary copies of the reconstructed public
data and checks that numerically equal but bitwise different client/reference
outputs are rejected:

```text
python -I -S -B test_replay_signed_zero.py --cpu-dir <cpu-extraction>/data --gpu-dir <gpu-extraction>/data --manifest <source-checkout>/experiments/short_cpu_service_reference_inputs_v1.json --scratch-dir <existing-absolute-scratch-directory>
```

The original candidate, owner/supervisor and outer-controller
receipts contain private process or path identifiers and are not published.
Their hashes in the saved result and independent reports are commitments to
those private bytes, not independently replayable process identity.
