# Fixed serial-service shared-load comparison: CPU rehearsal

This is a CPU-tested measurement core, not a GPU result or an enabled live
service benchmark. The live entry fails before starting an owner. The fixed
two-token service remains gated on its own source-specific load/cold/warm GPU
profiles, host admission, supervisor, marker and teardown checks. The generic
core does not provide those checks and is limited to a caller-owned synthetic
CPU service and explicitly synthetic memory values. This is a trusted-caller
contract: the provider's label is not authenticated, and the callable client
factory cannot prevent a caller from wrapping an existing live service. The
separate `run_live_service` entry always raises before taking action.

## Fixed workload

One resident owner serves the public input IDs `[15496, 995]`, shape 1×2,
seed 29, float32 output, no trace and no requested cancellation. Two distinct
teams each issue one warmup request, followed by two windows in fixed order:

1. **Serial:** one client thread submits four closed-loop requests for the
   first team, then four for the second. Only one request is outstanding.
2. **Shared:** two client threads start together, each submitting four
   closed-loop requests for its own team. At most two requests are outstanding.

The same inputs, seed and per-team counts are used in both windows. The plan
and its digest are recorded; client scheduling and actual admissions can
differ. The fixed order is a small development comparison, not a randomized
or steady-state study. A single execution owner serializes the worker calls;
two clients do not establish batching, simultaneous GPU forwards or fairness
under a sustained backlog.

Only successful responses with the expected team, broker and job identity and
a finite, exact float32 1×2×50257 output contribute to throughput. Every
warmup and measured result must use one broker, distinct job IDs and matching
per-team/ordinal output digests across the two windows for a valid pair.
Results retain digests and metadata, not logits, endpoints or credentials.
Each success represents two processed input positions; it does not generate
two output tokens. A window's completed requests and completed input positions
are divided by its common elapsed time. Per-team rates use that same window
denominator. A shared/serial rate ratio is descriptive and is not a causal
speedup claim or a confidence interval.

The controller uses `perf_counter_ns` for the window and client attempt/return
times. Broker queue time is `started_ns - ready_ns`; it excludes IPC connection,
authentication and input preparation before the broker's ready stamp. Service
time is `finished_ns - started_ns` and includes worker execution, synchronization,
CPU result transfer and broker completion, not just GPU kernels. Client response
includes socket and payload validation work. Each distribution reports count,
midpoint median, nearest-rank p95 and maximum; for eight successes p95 is the
maximum and is not a reliable tail estimate.

## Memory and lifecycle limits

The rehearsal accepts only a provider labelled `synthetic_cpu`, validates its
device-total/free and process-allocated/reserved/peak fields, and records
before/after values and the enclosing controller call times for each measured
window. Those values are fixtures, not device observations. A real run needs a separately reviewed owner-side provider
that samples the exact leased GPU's free/total and the owner's PyTorch
allocated/reserved and window peak counters under the actual owner guards. It
must bind its source, runtime, physical device, timestamps and failure behavior
to the experiment. The existing serial resource-profile samples are separate
observations and cannot be relabelled as memory measured during shared load.
Device-wide free snapshots and allocator counters do not bound other processes,
native allocations or transient peaks.

The core never starts, stops or attests the owner. The caller must retain the
prepared service, run, handles, threads and failure graph through clean
supervisor exit and marker reconciliation. A failed submit, rejection,
timeout, invalid output or unfinished thread stops further planned work and
invalidates the comparison; no request is retried. Client drainage stops new
submissions, releases any rooted start gate and joins known client threads; it
is not owner teardown. A false `requires_owner_reconciliation` means the client core
detected no listed fault; it never substitutes for supervisor and marker
checks. Bootstrap/finalizer faults and arbitrary repeated interruption remain
outside the tested cooperative thread model.

Before enabling a real run, independently review a bounded supervised driver,
the longer two-client host-memory envelope and the owner-side memory provider;
freeze their source and plan, obtain matching worker profiles, check fresh host
and device headroom and acquire a separately coordinated slot. If any package
source in the owner changes for instrumentation, its source fingerprint and
worker profiles change as well. The earlier two-client smoke and grouped pilot
do not certify this workload. This protocol must remain separate from isolated
tap or model-latency measurements.
