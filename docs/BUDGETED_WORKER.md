# Budgeted resident-model worker

Status: a serial, in-process integration of `MemoryBudget`, `ResidentWorker`, and the broker worker interface. CPU and simulated-failure tests establish the coordination behavior; this is **not** measured GPU capacity, allocator enforcement, a model loader, a process supervisor, a trace capture service, or a concurrency benchmark.

## Trusted configuration and profiles

The service constructs one `DeviceAdmission` per device and routes every participating worker on that device through the same object. It supplies one `MemoryBudget` whose telemetry callback reads that device, its explicitly indexed device name (`cuda:0`, for example), and a hard maximum incremental estimate. There is no cross-process lock or enforcement against another gate, a direct `MemoryBudget` user, other CUDA threads, or other applications. If any of those bypass the gate, its quarantine cannot stop their work. The owner must arrange exclusive process-level coordination and terminate/replace the worker process after quarantine.

Each `BudgetedWorker` wraps one already-created `ResidentWorker` and receives a nonempty immutable collection of service-owned `WorkloadProfile` records. A record is keyed by:

- A profile ID and a 64-character lowercase SHA-256 model/config/runtime fingerprint matching `worker.model_id`.
- The resolved device and exact batch/sequence shape.
- The sorted tuple of input names and dtypes; supported names and dtypes match `ResidentWorker`.
- The resident worker's per-request output limit.

It also supplies an estimated incremental byte envelope, which must be positive, must be at least the configured input plus output limits, and must not exceed the service hard estimate cap. Duplicate signatures or profile IDs, invalid bounds, wrong device or fingerprint, and incompatible worker caps are rejected. These checks catch obvious configuration errors; they cannot prove that an estimate is large enough. Neither the profile ID nor a correctly formatted fingerprint proves that a measurement occurred or that the loaded model matches its claimed identity. The trusted loader must verify the actual checkpoint/configuration/runtime and bind that verified fingerprint to `worker.model_id` and the matching profile.

Profiles must conservatively cover all allowed input values and seeds for the exact model, shape, names and dtypes, including input/output storage, activations, model-specific workspaces, allocator/runtime transients, backend/kernel selection and any permitted state. An observed peak on a few examples is not automatically such an envelope. Save the profiling protocol, measurements, environment, fingerprint and uncertainty separately; choose an explicit reserve and hard cap. The worker prohibits its supported hook-based tracing, and the wrapper requests **zero** GPU and host trace claims. It does not intercept arbitrary allocations or make profiling an allocation hard limit. OOM remains possible after admission, including when other applications use the device.

Only the fixed server configuration supplies profiles and estimates. A `ForwardRequest` supplies token tensors and a seed, not an estimate. The wrapper snapshots the request in CPU memory before profile selection and selects by the snapshot's shape and dtypes. It rejects requests without an exact profile rather than guessing, silently changing the shape, or using a client estimate. The resident worker takes another snapshot before execution. Callers must not mutate an input during snapshotting.

The wrapper captures the worker ID, device, batch/sequence/input/output limits and admission budget/device/hard cap. It rechecks them before snapshotting and after waiting for the gate; a mismatch prevents new execution. A mismatch observed after execution retains the lease and quarantines the gate. The owner must still keep configuration and model access exclusive: checking mutable attributes cannot prevent an unsynchronized thread from changing them between checks. `ResidentWorker` checks some model-state changes, but does not verify checkpoint bytes or sandbox model code.

The wrapper rejects ambient CPU or CUDA autocast on the execution thread before admission and before the forward. If a model exposes `config.use_cache`, the flag must be explicitly `False` at construction and remain unchanged. Models without this flag may still cache internally; the trusted loader must verify that their actual behavior is consistent with the profile. Neither check can prevent model code from transiently changing its own execution settings during a forward.

## Execution and failure handling

After a profile matches, the worker waits for the shared admission gate, checks cancellation and binding again, samples live device memory through `MemoryBudget.try_acquire`, and holds the resulting lease while the resident worker runs, offloads its result to CPU and synchronizes the device. A cancelled waiter can exit without waiting for another forward to finish. Cancellation of an already launched forward remains cooperative; the lease stays active until the resident worker returns after its drain. A failed admission runs no model work.

An explicit no-capacity return from `MemoryBudget.try_acquire` becomes an
ordinary `AdmissionError` rejection. Any exception during acquisition,
including an `AdmissionError`, leaves publication or lock state potentially
unknown: the wrapper latches the device quarantine without reacquiring the
budget lock and requires process replacement. The owner must not infer that a
lease was absent merely because the call did not return a handle.
The admission's pre-existing sticky flag is set before its best-effort event
notification, so a notification allocation failure does not clear the health
signal or allow another wrapper to proceed.

A peer can latch the shared admission quarantine while a forward is in
progress. If the owner observes the latch before releasing its lease, it
retains that lease and does not return a result. If it observes the latch just
after verified release, it suppresses the result but the lease is already
released. These checks are not atomic with a later broker publication: the
supervisor and broker must separately observe quarantine and suppress or
discard work as appropriate. An arbitrary asynchronous interruption around
the checks still requires watchdog-driven process replacement.

The admission gate records an in-flight marker before acquisition and clears
it only after a no-capacity receipt or verified lease release. Gate exit and
re-entry fail closed if that marker remains. Health reads do not wait on the
gate: they inspect a marker only when a nonblocking gate acquisition succeeds,
so an ordinary active forward is not reported as quarantined. Arbitrary
asynchronous interruption can strand an interpreter lock even around cleanup;
an unresponsive owner must be treated as uncertain by an external watchdog
and replaced rather than assumed safe to reuse.

On a valid successful return or a documented resident-worker failure known to occur before execution or after a successful drain, the wrapper releases its lease. An OOM after a verified drain is surfaced and poisons that worker, while its safely drained incremental lease is released. A failed synchronization, a set or unreadable worker process-quarantine latch, an unexpected exception, changed binding after execution, a result that violates the worker contract, or failed lease cleanup quarantines the shared gate and retains the lease. When cleanup fails after another failure, the visible error includes both. Quarantine is set before attempting to preserve extra references, since retaining them can itself fail under memory pressure. The budget independently retains its active leases. There is deliberately no in-process reset: replace the process after failure, and preserve a failure record. A dropped model reference or a cancellation request is not proof of device completion.

The gate serializes execution across wrappers sharing it. Its lock does not provide global per-team fairness: route all teams through a single broker and its fair queue, or add a separate fair scheduler before claiming global fairness. This integration does not batch independent requests or claim overlapping GPU execution.

For supervision, `worker.failed` is a nonblocking failure signal that remains false after a normal close. `worker.poisoned` also becomes true on normal close, because the broker must reject further submissions; `worker.quarantined` separately indicates that the execution process must be replaced.

The worker returns detached, per-request bounded CPU logits. Ownership transfers to the broker/client on success. The active lease covers device execution and not an unlimited collection of results retained by clients. Neither this wrapper nor the in-process broker bounds aggregate retained CPU logits; a serving transport must separately cap client count, response storage, unconsumed delivery time and backpressure. Trace storage is not supported in this worker path.

## Integration outline

The following is a shape of the service-side setup, not an executable or measured hardware profile:

```python
from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
from looking_glass.memory_budget import MemoryBudget, cuda_memory_snapshot

# The trusted loader has already verified its model/config/runtime fingerprint,
# loaded its model, and constructed `resident` with that fingerprint as model_id.
# A prior controlled protocol supplied the conservative `estimated_bytes`.
budget = MemoryBudget(lambda: cuda_memory_snapshot(0),
                      max_gpu_trace_bytes=0, max_host_trace_bytes=0)
gate = DeviceAdmission(budget, device='cuda:0',
                       max_estimated_incremental_bytes=service_limit_bytes)
profile = WorkloadProfile(
    profile_id=profile_revision, model_id=verified_model_fingerprint,
    device='cuda:0', shape=(1, 8),
    input_dtypes=(('input_ids', 'torch.int64'),),
    output_limit_bytes=resident.max_output_bytes,
    estimated_incremental_bytes=estimated_bytes,
)
worker = BudgetedWorker(resident, gate, (profile,))
# Broker(worker, ...) can now accept matching ForwardRequest instances.
```

Weights and any model-loading transients must be independently admitted and tracked by the owner before this setup; this wrapper begins with an already resident model. Live telemetry sees whatever is resident at each admission, but does not retroactively bound model loading. Device-wide free memory is a changing snapshot, not reserved capacity. No GPU experiment is implied by the example.
