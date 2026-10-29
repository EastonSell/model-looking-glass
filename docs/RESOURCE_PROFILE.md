# Trusted workload resource profiles

Status: this module provides an exact-match profile contract, a guarded
measurement helper, and a narrow adapter for the serial forward worker. It is
not yet a complete service, model loader, profile approval system, or GPU
benchmark result. **No profile in this document is a measured result, and no
estimated reservation guarantees that a workload will fit.**

## Trust boundary

Only a trusted service administrator or supervisor may create the profile
registry, choose its independently approved profile IDs, and construct live
workload keys. A request may select only an already supported operation and
input; it cannot supply an estimate, profile record, model identity, source
manifest, trusted profile ID, or device identity as authority. The supervisor
must derive the live key from the model and environment it actually controls,
including independently verified checkpoint contents and configuration,
source bundle, runtime and device mapping. Exact lookup rejects a changed or
absent key. The module does not establish that any particular caller or file is
trustworthy, does not prevent arbitrary Python code in the service process
from fabricating measurements, and does not make a model safe to execute.

Version 1 supports `model_load` and `forward` profiles. Generation, backward,
optimizer, and intervention workloads need separate operation-specific schemas
and bounds before they can use this contract. Each forward key binds:

- The service model ID, checkpoint content and revision, model configuration,
  source-code digest and a digest of the full runtime settings manifest.
- PyTorch, Transformers, CUDA runtime, driver and Python versions; explicit
  CUDA ordinal and complete canonical physical GPU UUID, device name, compute
  capability and total bytes; model/compute dtype and attention backend.
- Model and gradient mode, autocast, cache and compilation flags, cold/warm
  state and warmup-protocol hash, operation, capture mode and complete capture
  policy digest.
- The exact batch size, sequence length, sorted input names and dtypes, output
  cap and GPU/host retained-trace caps. These sizes are also the maximum sizes
  covered by that profile; a different size requires a separate profile.

The v1 input schema describes tensors of shape `(batch_size, sequence_length)`;
other ranks or input shapes require a new schema. A smaller shape is not
implicitly covered by a larger one: allocator behavior, selected kernels and
value-dependent model paths need not be monotone. Model-load keys have no
request shape, inputs, output or capture. Their estimate is separately resolved
and admitted **before** copying weights to CUDA.

`model_runtime_fingerprint` hashes the declared model/runtime identity using
canonical JSON with sorted string keys and deterministic separators. Its inputs
must be strings, integers, booleans, null, arrays or objects; floats are
rejected and should be represented explicitly in a stable settings manifest.
The `model_id` in `WorkloadKey` must equal that digest. The full workload key
has a second fingerprint that also includes operation, capture, exact shapes,
limits and warmup state. Every `MemorySample` contains that full fingerprint,
so a sample cannot be accidentally relabeled for another key. A profile ID
hashes the complete profile, including samples, evidence reference, provenance
and safety margin.

These hashes detect inconsistent bytes, not false attestations. In particular,
verifying the hash of a source manifest does **not** prove that the manifest
actually describes the loaded weights, configuration or running code. The
supervisor must verify those relationships independently and keep the approved
profile IDs in a separate trusted configuration. Likewise, the collector
records a supplied case digest and seed; it does not calculate the input hash,
set RNGs or prove that an input was used. The experiment must do those things
and preserve its source, configuration and raw evidence.

## Measurement and estimate

For a valid completed sample, the helper synchronizes the selected device,
reads the device-wide free/total and this process's allocated/reserved bytes,
rechecks headroom, resets PyTorch's peak counters, executes the callback,
synchronizes again, and reads ending memory and PyTorch's allocated/reserved
peaks. It requires a live supervised `OwnerSession`, matching cooperative
`DeviceLease` and active positive `MemoryLease`, with declared input/output/trace
allowances and capture caps covered by that lease. It checks the owner marker
before runtime or budget telemetry, and rechecks immediately before work. It
does not release either lease. The callback must complete its promised offload
and cleanup and return `None`.

For sample `i`, let `A` and `P` be PyTorch allocated and reserved bytes. Let
`F` be device-wide free memory. The observed incremental proxy is:

```
D_i = max(0, peak_A_i - A_before_i,
             peak_P_i - P_before_i, F_before_i - F_after_i)
U   = input_bytes + output_cap_bytes + gpu_trace_cap_bytes
B   = max_i(D_i) + U
estimated_incremental_bytes = B + margin_bytes + ceil(B * margin_bps / 10000)
```

`margin_bytes` must be positive; `margin_bps` is a nonnegative integer. The
input/output/trace allowance is added even when a measured peak already
included some of it. This intentional double counting and the explicit margin
are a cautious heuristic. They do **not** prove an upper bound: unseen values,
operator workspaces, lazy initialization, allocator fragmentation, external
pressure, or an unobserved combination can still cause an OOM. Host trace caps
are separate and do not account for all host memory.

Allocated and reserved peaks are PyTorch allocator high-water marks after the
reset; reserved includes allocator-managed cache and is not a separate amount
to add to allocated. The free/total values are device-wide endpoint samples,
not a device-wide peak. External applications can alter those endpoints, and
allocations outside PyTorch may be absent from its allocator peaks. A snapshot
can become stale immediately. See PyTorch's [CUDA memory semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
and [memory profiler limitations](https://docs.pytorch.org/docs/stable/torch_cuda_memory).

Provenance records an explicitly cooperative exclusive window, synchronized
boundaries with a peak reset, UTC collection time, protocol version, source
manifest digest and evidence digest. These are trusted attestations, not proof
that other non-cooperating applications were idle. Only completed samples are
accepted. A failed or cancelled callback yields no sample; the helper attempts
to synchronize after a callback failure and preserves the original error when
that succeeds. If synchronization fails, `ProfilingSynchronizationError`
retains the original callback error where applicable. The supervisor must keep
leases committed and quarantine the process/device until completion is proven;
it must never assume a failure or cancellation freed memory.

The helper requires the live, identity-aware `OwnerSession` created by the
supervised child, a UUID-derived `DeviceIdentity` and its matching live
`DeviceLease`. It checks the marker, parent watchdog, paths, visibility and
ordinal before querying the runtime, and verifies the runtime UUID and device
count before and after the callback. Its probe must obtain that UUID and count
from the actual CUDA runtime; echoing the key or a preflight UUID is not
verification. The trusted supervisor must provide the authoritative callback,
verify the common workspace/lock-path policy, and bind the memory budget's
telemetry to the same physical device. Four numbers in a `CudaMemorySnapshot`
cannot prove that binding. The checks cannot prevent non-cooperating
applications from allocating after a sample.

The owner must run `OwnerSession.check(device_lease)` and verify the actual
runtime binding before creating or acquiring a memory lease whose telemetry may
initialize CUDA. `collect_sample` checks its own entry, but cannot retroactively
protect CUDA calls made by the caller before the helper was invoked. The
supervised parent retains its unsafe marker until the child has safely drained,
synchronized, released or quarantined resource commitments, and attested clean
shutdown under the same device lease. A synthesized session or marker used by a
CPU unit test is not production supervision; see [supervisor protocol](SUPERVISOR.md).

## Persistence and worker bridge

`encode_evidence(key, samples)` produces canonical evidence bytes containing the
full key and completed sample records. A profile stores the evidence digest;
its constructor verifies that the samples and key produce that digest.
`load_profile` requires the saved evidence bytes, source-manifest bytes, and an
independently approved full profile ID. It rejects changed bytes, duplicate or
unknown JSON fields, mismatched identity, malformed telemetry and altered
margins. The application is responsible for writing the artifacts atomically,
never overwriting an existing run, retaining failure records and securing the
approved IDs and files.

```python
from looking_glass.resource_profile import load_profile, ProfileRegistry, resolve_worker_profile

# These bytes and the pin come from separately controlled service configuration.
profile = load_profile(profile_json, evidence=evidence_bytes,
                       source_manifest=source_manifest_bytes,
                       expected_profile_id=approved_profile_id)
registry = ProfileRegistry((profile,), approved_profile_ids=(approved_profile_id,))

# Construct this from verified live loader/runtime/device state, not a request.
worker_profile = resolve_worker_profile(registry, trusted_live_key)
```

The bridge first resolves the complete exact key, then permits only the serial
worker's eval/inference, no-autocast, no-cache, eager, forward/no-capture input
contract. It returns the worker's minimal `WorkloadProfile`; it does not verify
the supervisor's live key. The supervisor must bind the profile, warmup state,
model lifetime and worker configuration throughout service use. Model loading
needs its own exact load profile and lease; the forward bridge does not perform
or authorize loading.

The helper and contract have CPU-only tests using real cooperative file locks
and fake CUDA telemetry. No tests establish real-device memory envelopes,
maximum safe batch sizes, throughput or protection against all OOMs.
