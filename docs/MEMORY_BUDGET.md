# Memory admission and trace accounting

Status: a process-local policy and accounting primitive. It does not itself
schedule work, intercept CUDA allocations, validate workload estimates, enforce
trace caps in `TraceSession`, or guarantee that an admitted job fits. A broker
must explicitly integrate it before describing GPU admission or shared trace
caps as enforced.

## Device telemetry and admission

`cuda_memory_snapshot(device)` samples four distinct values: device-wide free
and total bytes from CUDA, and bytes allocated and reserved by this process's
PyTorch allocator. Device-wide free reflects other applications and device
allocations beyond this process's PyTorch allocator. Allocated is part of
reserved, not additional free capacity. These are sequential API reads, not an
atomic device snapshot. An unavailable, failed or invalid CUDA sample raises
`TelemetryError`. A custom telemetry provider must return a valid
`CudaMemorySnapshot`; a provider error or invalid return prevents admission.

For each admission, let `R` be the largest of 1 GiB, 15% of reported total
(rounded up to the nearest byte), and the configured `minimum_reserve_bytes`.
Let `C` be the sum of every active lease's incremental estimate plus the new
request's estimate. Both conditions must hold:

```
process_reserved + C + R <= device_total
device_free                  >= C + R
```

The second condition accounts for external device use; the first applies the
research allocation budget even if separate samples are inconsistent. The
entire estimate stays committed until its lease is released. As allocations
materialize, their bytes can appear in both process-reserved/device-free
telemetry and the estimate. This deliberate double counting is pessimistic and
may reject jobs that would fit. No cached allocator bytes are added back to
reported free memory. `BudgetSnapshot.admissible_estimate_bytes` is the
instantaneous upper bound on a *new estimate under these two policy checks*;
it is not physical free VRAM or an allocation forecast.

The broker must derive or validate estimates using trusted workload profiles and
request shapes. Estimates must include incremental input/output and retained
GPU trace storage, activations, scratch space, allocator/runtime transients and
any other expected peak allocation. The per-lease GPU trace cap cannot exceed
the estimate. Weights that are already resident must be accounted for in
telemetry; a lease acquired before loading a resident model must remain active
until that model is evicted and device work is complete. The default reserve is
a conservative starting policy, not a measured optimum.

`MemoryBudget.snapshot()` returns current telemetry, computed reserve, active
estimated commitments, the number of leases, retained trace claims and caps.
The sample can become stale immediately, external processes can allocate after
it is read, allocator fragmentation and unmodeled workspace can matter, and
CUDA out-of-memory is still possible. A snapshot is not a measurement of peak
memory. Admission itself launches no model work and performs no CUDA
synchronization.

## Trace claims and lease lifecycle

The policy has independent aggregate GPU and host retained-trace caps. Each
lease also has explicit per-lease caps. A trusted worker must call
`lease.reserve_trace('gpu' | 'host', byte_count)` **before** retaining a trace
buffer, and call `release_trace` only after that storage is no longer owned and
all work using it has completed. Claiming bytes is bookkeeping, not an
allocation, and allocation can still fail; a failed allocation must unwind its
claim after any work using it has been cleaned up. At GPU-to-host offload, both
claims may be needed while both copies coexist. These caps cover claimed
retained trace storage, not allocator peaks, ephemeral gradient workspaces,
pinned-memory pools, Python objects or unrelated host/GPU allocations. A
capture path that does not call the claim API is not constrained by it.

A lease is acquired before the allocation or work it covers, and remains active
until CUDA work has completed, offload is complete, references and trace
storage are disposed, and trace claims have been released. `release()` is
idempotent after successful release, but it raises `TraceClaimsOutstanding`
without changing accounting if any trace claims remain. A `with lease:` block
releases on normal exit; callers must ensure completed work, storage disposal
and claim release before leaving that block. On normal
exit with unreleased claims the context raises `TraceClaimsOutstanding`. On
any exceptional exit it preserves the original exception type and leaves the
lease active even if no trace claims remain, because device completion may be
uncertain. On Python 3.11+ it adds a note describing the outstanding cleanup.
The caller must explicitly release it after completion is established. Callers
can inspect `lease.released` and `lease.trace_bytes(location)` and must retain
or recover the lease for cleanup.

The budget retains active lease objects, accessible through
`budget.active_leases()`, even if a caller loses its own reference. The broker
must associate each lease with its job and owner; the ledger cannot infer
whether tensors, aliases, output artifacts, DMA transfers or asynchronous CUDA
work still exist. If synchronization or cleanup cannot establish completion,
keep the lease committed and quarantine the worker/device until cleanup is
verified or the owning process is safely replaced. Cancellation, an exception,
a dropped model reference, and worker poisoning are not proof of completion.

The lock protects operations on one budget instance in one process only. It
is not an interprocess lock or a reservation against unrelated applications.
Multiple independent budget instances for the same GPU can over-admit. A single
trusted broker must own the budget and route compatible jobs through it.

## Minimal policy setup

```python
from looking_glass.memory_budget import MemoryBudget, cuda_memory_snapshot

budget = MemoryBudget(
    lambda: cuda_memory_snapshot(0),
    max_gpu_trace_bytes=64 * 1024**2,
    max_host_trace_bytes=128 * 1024**2,
)

# Obtained from a trusted profile for this exact bounded workload.
lease = budget.acquire(
    estimated_incremental_bytes=256 * 1024**2,
    max_gpu_trace_bytes=32 * 1024**2,
    max_host_trace_bytes=64 * 1024**2,
)
# Execute and track buffers through this lease. Complete/synchronize work,
# dispose or transfer owned buffers, and release all trace claims before:
lease.release()
```

This example reserves nothing on the GPU and does not execute a workload. The
ledger and its CPU tests use mocked telemetry; measured GPU headroom and
reliable workload estimates must come from separate, controlled experiments.
