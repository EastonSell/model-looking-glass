# Isolated CUDA trace resource calibration, proposal v1

Status: **proposed protocol, not implemented or run**. No resource estimate in
this document is a measured limit, a guarantee against out-of-memory (OOM), or
authorization to enable the CUDA benchmark. It defines a small, independently
reviewable pilot for the exact workload in `CUDA_BENCHMARK_V1.md`. The existing
two-token, no-capture service bootstrap has a different execution path, shape,
gradient mode and allocator history. Its profiles cannot authorize any stage
here. Run no GPU work until the required implementation and independent review
below exist and the shared GPU coordinator schedules an isolated window.

## Workload and scope

Use the offline loader's complete verified seven-asset manifest for
`distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
The `model.safetensors` SHA-256 is
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The actual source, asset and runtime identities must match independently
reconstructed manifests. No model download or remote code is permitted.

The pilot binds batch 1 and eight `torch.int64` IDs
`[15496,11,616,1438,318,257,1332,13]`, FP32 eager evaluation,
`torch.no_grad()`, no autocast, KV cache, compile, intervention or generation.
For all six layers, capture block, attention and MLP outputs with sample 0 and
token positions 0, 3 and 7, on the source GPU. Require exact path, role,
layer, output selector, forward index and invocation metadata. The expected
value modes retain 18 `(1,3,768)` FP32 tensors, totaling 165,888 logical
bytes. Each session is capped at 32 events and 4 MiB of retained GPU tensor
payload; those caps do not bound allocator rounding, temporary tensors, Python
metadata, model or workspace. ID mode saves no values. `none` installs no
session. Preflight requires matching dtype, shape and device, then compares
all four outputs and direct/gradient captured values with `torch.equal` and no
tolerance. This is numeric equality under that operator, not a bit-identity
claim: signed zeros can compare equal. GPU subnormal behavior is not yet
established for this workload. `torch.equal` also accepts matching infinities;
the merged benchmark preflight separately checks finite logits for the
baseline and every mode. Include those reductions and synchronization in the
profiled composite preflight. Later timed outputs do not have the same finite
check, so the preflight alone does not establish their finiteness.

Record separate stages for model load, **persistent setup plus combined
equivalence preflight**, and each of `none`, `id`, `direct` and `gradient`.
The setup sample begins before creating the CUDA input tensor, includes the
actual equivalence check, and ends after the synchronized CUDA-event creation
and priming used by the benchmark. The input and event objects stay live for
later modes under their still-held setup commitment. The existing equivalence
preflight retains the baseline, successive logits and both direct and gradient
sessions at once, and constructs gradient-reporting temporaries while the
direct values remain live. It therefore is not a single-mode forward. A
single-mode profile, including the larger of two such profiles, cannot by
itself admit the preflight. Before enabling, either implement a versioned
composite workload key and profile for the actual preflight or redesign the
preflight with reviewed evidence and reprofile the changed behavior. Do not
mislabel it as a direct or gradient-only `resource-profile-v1` workload: the
current `WorkloadKey` supports only single capture modes and `model_load` or
`forward` operations. For planning, the composite allowance must include
two independently capped 4 MiB GPU trace sets, at least three simultaneously
live 1,608,224-byte FP32 logits tensors, input storage, value-comparison and
finite-check temporaries, autograd workspace, and allocator/context overhead.
These allowances are conservative declared payloads, not measured peaks.
Revalidate lifetimes against the exact source selected for the run.

The initial pilot is the benchmark's **quick** schedule only: ordering seed 17,
one warmup round and three measured rounds, shuffling all four modes once per
round with the benchmark's actual schedule function and warmup seed transform
`0x5A17`. Profile the warmup calls as well as measured calls, in their actual
order, after the combined preflight and the benchmark's actual synchronized
CUDA-event creation and priming, without `empty_cache()` between calls. Record
and replay every intervening allocation-affecting setup and monitor policy. This
captures observed allocator-cache transitions rather than treating four modes
as independent cold processes. Record every actual mode, round, position,
phase and completed sample. Preserve the fixed PyTorch seed and schedule seeds.
One input, device and process cannot establish a general resource bound. A
full four-seed, 10-warmup/30-measured-round protocol needs its own reviewed
calibration and schedule identity; quick evidence cannot silently authorize it.

## Identity and admission

The calibrated runner and later benchmark must use the same frozen source
snapshot, model assets, CUDA-visible physical GPU and actual runtime settings.
Fingerprint the exact benchmark and collector code, mode-specific capture
policy including selectors and caps, complete token case, preflight, phase
schedule and warmup history. Bind the actual case digest into the trusted
workload identity, not merely into sample metadata: the current
`WorkloadKey` has no input-value field. A versioned key or a reviewed,
canonically reconstructed policy manifest can provide that binding. Require
each sample's case digest to match the key and the actual input. No client or
candidate profile may supply the trusted identity or a memory estimate.

Preflight and owner must enforce an explicit, nonsecret child environment
through the real supervisor's **replace-environment** facility, not an API
that merges omitted parent variables. Reject unlisted CUDA, PyTorch, TF32,
allocator, dynamic-loader and thread settings. Fix and record
`CUDA_VISIBLE_DEVICES` to the selected full UUID, `CUDA_DEVICE_ORDER` to
`PCI_BUS_ID`, `PYTORCH_CUDA_ALLOC_CONF` to `backend:native`,
`CUBLAS_WORKSPACE_CONFIG` to `:4096:8`, offline Hub settings and one-thread
OMP/MKL settings; independently verify the actual allocator and all runtime
flags before work. Apply identical values during later benchmark use. The
frozen settings include deterministic algorithms on, cuDNN benchmark off,
cuDNN deterministic on, cuDNN and matmul TF32 off,
`torch.manual_seed(20261028)`, and one PyTorch intra-op thread; set and verify
the same supported inter-op and other backend/thread controls in both runs.
The isolated benchmark's current scaffold computes a larger allocator
fraction from its reserve rather than setting 0.20. Switching to this proposed
cap and reading it back therefore requires a separately reviewed code change
before calibration or use.

The benchmark's source-first `-I -S -B` startup with a fresh, nonexistent bytecode
prefix ignores `PYTHONHASHSEED`; record that limitation rather than asserting
a fixed hash seed. It must also fail closed on untrusted fallback import
finders and verify allowed source and dependency origins before execution.
Before importing Torch on Windows, inventory the exact manifest-backed
`torch/lib` native DLL set and reject unlisted or changed files: Python's
import guard alone does not intercept `LoadLibraryExW`. OS, driver and other
native dependencies remain outside complete authentication; record the limit.
The implementation must verify these policies, not merely write them to a
manifest.

Use the canonical outer workspace root and the same physical-UUID-derived
supervisor marker and `DeviceLease` as all cooperating GPU owners. After the
lease and supervised owner are checked, verify that CUDA ordinal 0 actually
binds to that UUID and that exactly one physical GPU is visible. Inspect live
total and free VRAM; the previously observed RTX 2070 SUPER with 8,192 MiB is
identification, not an admission snapshot. Fail closed on unavailable or
inconsistent telemetry or identity. Hold the device lease through child exit.

For the pilot, propose a native PyTorch allocator fraction of `0.20`, set and
read back before model GPU allocation, and require the same fraction, native
backend and unchanged device total during later benchmark use. Verify the
effective byte-truncated limit rather than requiring the floating readback to
equal the decimal literal. This process allocator ceiling does not limit CUDA
context, driver, non-PyTorch or foreign allocations. Keep a device-wide reserve
`R = max(1 GiB, ceil(15% of total VRAM))`. Independently check both
`process_reserved + all_live_commitments + R <= total` and
`all_live_commitments + R <= device_free` immediately before each stage and
after initial synchronization, as well as the matching live owner and lease.
Stop on pressure; no automatic larger retry or precision substitution.

Provisional pilot commitments are 2 GiB for model load and a separate 1 GiB
for persistent setup plus the combined preflight. Acquire the setup commitment
before creating the CUDA input or event objects. Hold the load commitment
while any model aliases remain. Keep the setup commitment after preflight for
the CUDA input and event objects until their final synchronized disposal; do
not silently transfer their custody or assume callback return frees them. Each
sequential mode additionally holds a 1 GiB job commitment until that mode's
output and trace are disposed and synchronized. The live committed total is
3 GiB during setup/preflight and 4 GiB during a mode. Recheck the load, setup
and mode commitments plus the reserve against both inequalities before every
mode. Keep these commitments even after memory is realized, so the checks
deliberately double-count. The
setup/preflight lease must cover both 4 MiB GPU trace payload caps; a single
value-mode lease must cover its 4 MiB cap. The 2 GiB and 1 GiB amounts are
starting assumptions, **not** bounds derived from the two-token profile. If a
later reviewed estimate or a live budget requires more, refuse the workload
rather than run under a smaller
commitment. Set the owner's aggregate retained GPU trace cap to 8 MiB, reserve
each retained tensor's claim before allocating it, and release the claim only
after that tensor and its aliases are disposed. Each session still enforces its
own 4 MiB and 32-event limits. Temporary selection, validation and autograd
tensors also consume device memory but are not retained-trace claims; include
them in the stage commitment and observed allocator peaks. The existing local
session cap alone is not an integrated `MemoryBudget` claim. Trace metadata,
CPU model storage and other host allocations are
not covered by the GPU trace-byte limit; enforce separately reviewed host
claims and event/item limits, and do not call the host ledger a machine-wide
RAM cap. The memory manager's estimate and the allocator cap reduce exposure;
neither prevents every external or transient allocation.

## Measurement and profile construction

Within a supervised, cooperatively exclusive owner, collect one synchronized
load sample, one setup/combined-preflight sample, and every quick-schedule
warmup and measured mode sample. Use the actual workload and lifetimes: retain
each transient output and trace through the completion synchronization, dispose
them, then synchronize again. The setup/combined sample must include CUDA input
construction, the
actual complete equivalence check including simultaneous sessions, comparisons
and cleanup, and the subsequent CUDA-event creation and priming. The input
tensor and event objects remain in an explicitly held owner through the later
mode samples. Their still-live bytes are part of later allocator baselines;
event and driver allocations may not appear in PyTorch counters. A profiling
callback returning `None` is not evidence that these persistent objects were
disposed. No CPU transfer of trace tensors is introduced into the measured
workload. Include hook setup,
invocation tracking, position selection, finite reductions, gradient readout,
validation and hook removal. Check exact event counts, trace bytes and output
shape; the pinned vocabulary gives `(1,8,50257)` FP32 logits of 1,608,224
bytes. A shape or capture mismatch is a failed attempt, not a smaller profile.
Finite validation transfers a reduction scalar to the host and synchronizes;
the full captured tensors remain on the GPU. Add CPU adversarial tests for
the equality comparator's signed-zero and subnormal semantics, and report any
GPU-specific subnormal observations separately instead of assuming bitwise
preservation.

The existing benchmark discards warmup measurement rows and does not enforce
per-sample event counts or output shape. Retaining and checking every warmup
and sample therefore requires a reviewed refactor. Any added validation,
allocation or changed lifetime must be present in both the calibrated path
and the later benchmark path; otherwise their profiles do not match.

Immediately before each sample, synchronize, check live ownership, budget and
headroom, read device-wide free/total and process allocated/reserved bytes, and
reset process allocator peaks. After the operation and disposal synchronize
again, read process peak allocated and peak reserved and the endpoint values,
and recheck the lease. Store the adjacent admission snapshot, all commitments,
and the start/end timestamps. Endpoint free memory is not a peak. Peak
allocated and reserved are PyTorch process counters; reserved may include
cache from prior modes. They omit context, non-PyTorch and foreign memory.
Optional device-wide polling records sampled extrema and query intervals, not
the true peak; monitor overhead and external activity can affect results.

Do **not** blindly wrap the current `measure_sample` in `collect_sample`: both
reset the same CUDA peak counters. A nested reset can erase allocations inside
the outer interval. Refactor to one shared reset owner, or implement a
reviewed, equivalent single-reset collection path; test its ordering with a
fake device and verify that all trace/output lifetimes are inside the same
sample. Readback alone cannot prove equivalence. Timings from this calibration,
including collection and monitoring, are not latency or shared-throughput
results; use the separately scheduled benchmark for those claims.

For each supported exact key, preserve all completed raw samples and calculate
the existing profile's observable increment as the maximum, over its samples,
of zero, `peak_allocated - allocated_before`,
`peak_reserved - reserved_before`, and `free_before - free_after`. Add the
key's declared input, output and GPU trace allowances; then apply a predeclared
margin. For the proposed pilot use 512 MiB plus 25% of this basis for load,
and 256 MiB plus 25% for each single-mode forward. The composite preflight
needs its own versioned formula and explicit simultaneous allowances before a
candidate can be constructed. Record and independently recompute every term;
adding declared allowances may intentionally double-count observed storage.
Device-free endpoint deltas and finite measurements can miss transient external
allocations, and a margin cannot transform a sample into a guaranteed bound.
An estimate exceeding its provisional lease or live headroom fails the pilot;
do not claim that observing it proves the commitment safe.

Only a profile whose full source/runtime, actual case, exact operation, mode,
schedule and capture policy are independently reconstructed and match may be
considered for that exact workload. The owning team stores full reviewed
profile pins independently of the run directory. No profile may authorize a
different case, cap, precision, other mode, preflight, service request,
background workload or full benchmark schedule. Calibration and use remain
isolated from shared-load throughput work.

## Failure evidence and enablement

Use a new private run directory. Persist a start record and the proposed
protocol/source/runtime/asset/case/schedule manifests before device work.
For each completed stage, durably append its admission, synchronized memory
sample, trace and output checks, owner/run identity and status. Preserve partial
journals, telemetry query errors and a failure record on exception,
cancellation, OOM or failed synchronization. Never turn an incomplete stage
into a completed sample or publish a successful profile when any required
stage, validation, artifact hash or clean device teardown is missing.

Stop submissions on OOM or cancellation; keep claims charged until the
corresponding work and storage are verified disposed. Synchronize, release
model, output and trace aliases and inspect outstanding claims. The supervised
child attests clean while still alive and holding its lease, after it verifies
device drain. The supervisor then reaps that matching child and verifies the
attestation and exit before the parent publishes. A timeout, failed
synchronization or uncertain cleanup leaves the shared unsafe marker and
quarantine in place. A scientifically failed
calibration may have clean teardown; a successful device teardown does not
make the experiment successful. Persist these outcomes separately. Public
summaries omit physical UUIDs and sensitive paths; private manifests retain
identities required to verify the run.

Before any GPU run, independently review and CPU-test the composite identity,
exact case binding, profile estimator, admission math, interruption and OOM
cleanup, single-reset sample boundaries, full source/native-DLL and environment
guards, and parent-authoritative artifact publication. Integrate the resource
guard into every actual preflight and mode, including warmups; the current
benchmark's disabled scaffold does not yet do this. Confirm the collector and
later benchmark use the same allocator cap and runtime. The experiment team may
then request an isolated window from the shared coordinator, run only the
bounded quick pilot, and publish verified observations with their limitations.
