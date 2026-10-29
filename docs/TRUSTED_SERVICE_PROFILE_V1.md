# Trusted-process two-token service resource protocol v1

Status: **implementation preparation; no measured profile yet**. This protocol
supports one fixed DistilGPT2 worker path on one supervised device. It uses a
trusted owner process and a recorded source/runtime snapshot. It does not prove
which Python bytecode, native libraries, operating-system components or driver
were executed; the operator must keep the selected source and environment
stable and run the recorded code. Hashes, package versions, and recorded sample
fields establish consistency of evidence, not its authenticity. This protocol's
`trusted-service-source-v1` and `trusted-service-runtime-v1` identities differ
from the stricter bootstrap schemas: profiles, artifacts and audit results are
not interchangeable. A separate direct-model one-shot is not a worker profile.

The owner uses the pinned offline `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, verified seven-asset cache, float32,
eager/eval/inference, no cache, autocast, compile or capture. The only request
is contiguous CPU int64 `input_ids=[[15496,995]]`, shape `[1,2]`; no other
input field is permitted. Execute the actual `ResidentWorker.execute` path,
including request snapshot, RNG handling, device copies, output offload and
synchronization. Verify the full finite `[1,2,50257]` float32 result fits the
512 KiB tensor cap, and enforce the actual serialized response's 4 MiB cap.
Worker input is capped at 64 KiB. The one cold warmup uses seed 17; the three
warm measurements use seeds 29, 43 and 71 in that order. The initial service
accepts only those three warm seeds and this exact input. Broader inputs, seeds,
shapes, modes, batched requests and capture require a separate protocol/profile.
The trusted request factory validates decoded token values, constructs a fresh
private fixed CPU tensor, and retains exclusive ownership until the worker has
snapshotted it. Validating a caller-owned mutable tensor before a later copy
does not by itself establish which values the worker actually consumes.

`prepare_trusted_source` records hashes of the selected package Python files,
this protocol document, base interpreter executable and the seven pinned
offline assets. `build_trusted_service_identity` repeats the source/asset
preflight, records the actual imported Torch/Transformers top-level file hashes
and versions, and under a real supervised owner and its physical-device lease
verifies the actual runtime UUID, device count, total, driver, CUDA/PyTorch and
cuDNN versions, allocator, fixed math flags, thread limits and runtime
environment. The owner must launch profiling and service from the same stable
source snapshot and dependency environment; changed source or runtime creates a
different identity and needs new profiles. Paths and physical UUIDs in manifests
are private. The model/forward mode entries are planned contract values until
the actual model and worker execution are verified. Pre-copy verification must
validate pinned CPU tensors, configuration and same-object tied embeddings,
with overwrite/swap conversion flags false; after copy and synchronization,
verify the head and embedding remain the same device Parameter and storage.
If CUDA was initialized before the live identity helper runs, the trusted owner
must have performed that initialization under the same supervised lease; the
helper does not prove the history of an existing process context.

`HF_HOME` must be an explicit canonical existing directory strictly containing
the selected canonical offline cache root. Validate this before importing
Transformers, and include the exact `HF_HOME` in the runtime manifest used by
both collection and service. A missing, changed or noncontaining value fails
closed; previously collected identities without it do not match this runtime.
The replacement environment must also set `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`. Record these settings
and reject other `HF_`, `HUGGINGFACE_` and `TRANSFORMERS_` overrides. The
loader's offline asset checks remain a separate control.
The owner must start a fresh process without preloaded Hub or Transformers
modules. After the supervised identity is established and Transformers is
imported, the helper checks and records the effective import-time Hub and
Transformers cache paths, offline modes and telemetry setting against that
environment. These checks do not authenticate previously executed code.

Use an isolated cooperative device window, the common UUID-derived lease and
unsafe-marker supervisor. Establish ownership before every CUDA call and retain
the lease through child exit. Apply and read back the native PyTorch allocator's
0.20 fraction before copying weights, comparing the effective integer-byte
limit against the unchanged reported device total. The cap does not limit all
CUDA, driver, context or external-process allocations. Start with a separately
held 2 GiB model-load commitment plus a 512 MiB commitment for each sequential
forward, and trace caps of zero. Require reserve
`R=max(1 GiB, ceil(15% * device_total_bytes))` and both live checks:
`process_reserved + outstanding_commitments + R <= device_total`, and
`outstanding_commitments + R <= device_free`. Recheck after initial
synchronization and before work. Retain the load commitment while model aliases
remain; this deliberately double-counts realized allocations. The estimates
and allocator cap reduce exposure but do not prevent OOM or external pressure.

Collect one synchronized load sample with load-case digest and seed 17, one
cold forward sample with the fixed case digest and seed 17, then three warm
forward samples with the same case digest and seeds 29, 43, 71. Use one peak
reset after the synchronized baseline and synchronize again after each complete
worker operation and output offload. Record device-wide free/total separately
from PyTorch process allocated/reserved and their peaks; endpoint free values
are not device-wide peaks. Store raw samples, source/runtime/asset and case
manifests, exact software/device identity, ordered stages, failures and clean
teardown evidence. Preserve partial and failed attempts; no cancellation, OOM,
invalid output or failed synchronization becomes a completed sample.

Build the existing `resource-profile-v1` estimates from those exact samples.
Predeclare a 512 MiB plus 25% margin for load and 256 MiB plus 25% for each
forward; reject profiles whose resulting estimates exceed the respective
commitments. Require three independently configured full profile-ID pins, exact
live keys/source identity, and exactly the ordered cases and seeds above. These
case/seed fields are attestations; the owner must enforce actual inputs, RNG
handling and workload execution. A finite observation and margin do not prove
an upper bound. On OOM or uncertain drain, quarantine the owner and retain its
claims/marker until the supervised process is safely reaped; only verified
cleanup and a matching clean supervisor result permit normal reuse.
