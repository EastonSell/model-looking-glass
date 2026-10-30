# Experimental execution infrastructure

These components explore **cooperative ownership, bounded bookkeeping and
failure-aware handoff**. They are not a supported general GPU service or the
recommended starting point for observing a model. Start with the
[GPT-2 workflow](GPT2_TRACE_WORKFLOW.md) and [project scope](PROJECT_STATUS.md).

This guide consolidates the recurring contracts from the historical
broker/IPC/worker/bootstrap manuals. Detailed original contracts remain at the
[pre-consolidation documentation snapshot](https://github.com/EastonSell/model-looking-glass/tree/f3d3e28980379640f7548541225ef40b017e5557/docs).
That snapshot records development stages; statements such as “not yet wired”
are historical, not a current integration checklist. It is not an executable
support matrix. Use the exact source, tests, protocol and runtime for the
component or recorded study being investigated.

## Component map

| Component | Role and boundary | Source / focused tests |
|---|---|---|
| Resident broker | Many callers, one trusted worker executing serially; no automatic batching or concurrent GPU kernels | [broker](../looking_glass/broker.py), [tests](../tests/test_broker.py) |
| Local IPC | Loopback, forward-only transport; not traced jobs, public serving or multi-tenant isolation | [IPC](../looking_glass/ipc.py), [tests](../tests/test_ipc.py) |
| Device lease | Exclusive participation in one agreed lock-file protocol; does not control unrelated GPU users | [lease](../looking_glass/device_lease.py), [tests](../tests/test_device_lease.py) |
| Device budget | Telemetry-based estimated admission and retained-trace accounting; allocation can still fail | [budget](../looking_glass/memory_budget.py), [tests](../tests/test_memory_budget.py) |
| Host budget | Handle-based declared storage, item, request and operation commitments; not measured RSS | [ledger](../looking_glass/host_budget.py), [tests](../tests/test_host_budget.py) |
| Private CPU trace owner | One trusted traced worker and sealed batch; not itself broker/IPC/GPU integration | [owner](../looking_glass/cpu_trace_owner.py), [tests](../tests/test_cpu_trace_owner.py) |
| Sealed trace handoff | Exact job/recipient identity, metadata and payload acknowledgements; not a wire protocol | [handoff](../looking_glass/trace_batch_handoff.py), [tests](../tests/test_trace_batch_handoff.py) |
| Fair lifecycle seam | Coordinates a supplied bridge/server/broker graph and watchdog; does not construct or authorize a GPU service | [lifecycle](../looking_glass/fair_service_lifecycle.py), [tests](../tests/test_fair_service_lifecycle.py) |
| Fixed GPU owner | Source/profile-bound, narrow serialized workload; not a general model-serving entry point | [owner](../looking_glass/gpu_service.py), [tests](../tests/test_gpu_service.py) |
| Bootstrap and saved audit | Source/environment checks, owner records and artifact consistency; not independent execution attestation | [bootstrap](../looking_glass/gpu_bootstrap.py), [audit](../looking_glass/resource_bootstrap_audit.py), [bootstrap tests](../tests/test_gpu_bootstrap.py), [audit tests](../tests/test_resource_bootstrap_audit.py) |

A file's presence, passing unit tests, or a successful fixed historical pilot
is not evidence that the whole graph is integrated or safe for broader jobs.
In particular, CPU tests using fake CUDA dependencies are not GPU execution.

## Ownership and completion

- Keep one explicit owner for the model and its lifecycle. Callers must not
  mutate inputs during snapshots or access the worker/model behind its owner.
  Shape and identity checks do not prove unchanged code or weights.
- Acquire claims **before** allocating or retaining covered buffers. Keep both
  source and destination charged while a copy exists. A claim release records
  the owner's disposal attestation; it does not free a tensor or external alias.
- Cancellation blocks new work; it does not stop native work, free storage or
  establish completion. An exception, dropped reference, joined thread or
  poisoned worker is not sufficient evidence of a clean device drain.
- A call may commit before its return is lost. Reconcile the exact retained
  handle and generation before retrying. In particular, byte-count releases
  are not universally idempotent; blind retries can uncharge live storage.
- Unknown disposal, uncertain native completion or failed shutdown requires
  retained commitments and quarantine, not optimistic release. An external
  supervisor must bound stalled work and establish the relevant process/device
  recovery. Do not replace an unsafe owner with another broker in the same
  process and call that recovery.

These are cooperative Python contracts, not protection against hostile
same-process code. Arbitrary asynchronous interruption can strand locks or
leave unknown native resources. Ordinary cancellation and tested failure
paths do not establish safety at every bytecode boundary. The
[experimental opcode test policy](EXPERIMENTAL_OPCODE_TESTS.md) preserves that
qualification explicitly.

## Budgeting is not physical memory enforcement

`MemoryBudget` compares estimated commitments with sequential CUDA telemetry.
With reserve `R = max(1 GiB, ceil(0.15 * device_total), configured minimum)`
and total active-plus-new estimated commitments `C`, admission requires both:

```text
process_reserved + C + R <= device_total
device_free                  >= C + R
```

This deliberately conservative accounting can count an estimate alongside its
already materialized allocator use. Samples are not atomic and can become
stale immediately. External allocations, fragmentation and unmodeled workspace
can still cause OOM. Cached allocator bytes are not extra free capacity.

Host/GPU retained-trace caps cover only participating declared storage. They
do not include every Python object, transient, pinned-memory pool, socket
buffer or external process. `HostBudget` separately bounds items, requests and
operations; even a zero-byte claim consumes an item. An operation pins its
claims until the owner confirms the copy/serialization/send has stopped.
Independent per-process ledgers do not impose a machine-wide limit.

## Handoff and local transport

A sealed trace handoff identifies the exact job, generation, transfer,
recipient and event. Completion requires metadata and every payload to be
acknowledged. Acknowledgements are cooperative attestations, not remote
proof of receipt or disposal. A synchronous sink must retain no payload,
including through exception/traceback aliases; retaining or asynchronously
sending a copy requires a separate owner and claim.

The existing IPC transport uses loopback bearer credentials and trusted local
participants. Do not expose it through a proxy or tunnel, log tokens, or put
them in command-line arguments. Clients do not authenticate the server: a
reused port can expose a stale token to another local process. Distribute
fresh owner-lifetime endpoints/credentials privately and discard stale clients.
These limitations remain even when the request schema and size checks pass.

## Profiles, evidence and validation

Source hashes and saved audits establish consistency with recorded bytes,
not which machine code executed or whether an unobserved process completed.
Profiles apply to their exact source/runtime/workload identity; a changed
implementation is not automatically covered by an earlier measurement.
Missing checks stay `not_checked`, missing terminals stay incomplete, and a
later successful run does not erase a failed or quarantined predecessor.

For an actual historical replay, use its pinned protocol, source and runtime,
not this summary as launch authorization. The
[archive instructions](../archives/README.md#replaying-a-historical-study)
explain restoration without rewriting old hashes. The
[maintenance record](MAINTENANCE_2026-10-29.md) distinguishes executed CPU
coverage, experimental exclusions and unmet platform/runtime prerequisites.
