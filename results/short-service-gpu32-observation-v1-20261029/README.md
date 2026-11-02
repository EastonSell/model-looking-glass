# DistilGPT2 32-token GPU routing observation

One supervised development run used a source-configured single-resident
DistilGPT2 owner and recorded four serial model entries at shape `1 × 32`.
Two threads submitted client requests through an in-process Broker; the run
did not measure whether their submissions overlapped. Each client's complete
1,608,224 float32 logits was bit-identical to its own same-device serial
reference. The four saved full-logit files are 6,432,896 bytes each. The two
last-active reference rows
differed on 50,256 of 50,257 logits at the fixed tolerance
`1e-6 + 1e-5 * abs(reference)`, so swapping these two results would be detected
by that check.

The inputs were the all-active synthetic-32 row and the ligand row with 23
active tokens followed by nine right-padding tokens. Both were already public,
reserved and CPU-exposed. This is one routing and repeatability observation,
not a held-out test, IPC or trace observation, batched forward, throughput
benchmark or general service-capacity guarantee. A separate
[16-token observation](../short-service-gpu16-observation-v1-20261029/) used a
different pair and its own admission.

The separately published [CPU reference](../short-service-cpu-reference-v1-20261029/)
contains full logits for these padded inputs and an unpadded ligand reference.
In a post-run descriptive comparison, CPU and GPU full logits were not
bit-identical. The maximum absolute difference was `6.866455078125e-5` for
synthetic-32 and `9.1552734375e-5` for ligand; zero of the 1,608,224 logits
per row exceeded the routing tolerance when applied to the CPU values.
Last-active maximum absolute differences were `6.103515625e-5` and
`5.7220458984375e-5`, respectively, with zero violations. This cross-runtime
comparison was not a prespecified pass gate.

The 21 saved synchronized memory observations reported a minimum sampled
device-wide free value of 7,016,022,016 bytes (6,691 MiB). The largest
process-local reset-window peaks were 374,831,616 bytes (357.47 MiB) allocated
and 408,944,640 bytes (390 MiB) reserved. Free memory is sampled at points;
allocator peaks are process-local and do not include every native allocation
or other process. These measurements are not a bound for future requests.

The saved owner record says known model/input/output references were released
and process-local allocated/reserved counters and logical host and device
claims were zero after guarded cleanup. The supervisor and outer controller
records report clean exit 0 and a reaped direct controller. The coordinator
reported an unlocked OS-only post-exit marker and numeric-process check; no
separate durable receipt of that point was saved. These records do not prove
complete native CUDA teardown, descendant absence beyond the supervised scope,
global device idleness, or independently authenticate model execution. No
cancellation was exercised.

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
`archive-index.json` pins all parts and members. From this directory,
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
python -I -S -B replay_short_cpu_gpu_public.py --cpu-dir <cpu-extraction>/data --gpu-dir <gpu-extraction>/data --manifest <source-checkout>/experiments/short_cpu_service_reference_inputs_v1.json --source ca9f5a5e6d4593cfe16c4e1cfa19800f2d6df5e7 --bucket 32 --result-sha256 1f25e5d7b18c655428506c24990f28ca966d00bc73583b195b19aaa44f205e0b --output <new-absolute-report-path>
```

The replay reads saved logits and records, with no model or GPU call. Its
published output is `short-service-gpu32-public-replay-v1.json` (SHA-256
`63a995d8219e62a7dcc66b78f2c91be5820331f35199f9a5d3371af4a8896c12`).
The original candidate, owner/supervisor and outer-controller receipts contain
private process or path identifiers and are not published. Their hashes in
the saved result and independent reports are commitments to those private
bytes, not independently replayable process identity.
