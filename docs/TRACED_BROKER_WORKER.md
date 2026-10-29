# Trace plan binding for a broker job

`bind_trace_plan` validates and snapshots trace selections for **one** trusted
owner's job. It produces metadata for a future traced worker; it does not run
a model, own hooks, reserve storage, submit to the broker, deliver tensors, or
authenticate identities. The existing broker is forward-only: it can discard
a result during cancellation or error and has no trace-payload disposal
protocol. Do not submit trace payloads through it. This plan is deliberately
not a broker worker and does not implement its submission interface.

```python
from looking_glass.traced_broker_worker import (
    BrokerJobIdentity, SampleTrace, SiteTrace, TraceLimits, TraceSpec,
    bind_trace_plan,
)

identity = BrokerJobIdentity("a" * 32, 7, "team-a", "client-request")
spec = TraceSpec(
    run_id="run-unique-to-this-forward",  # assigned by the trusted owner
    samples=(SampleTrace("sample-0", (0, 2)),),
    sites=(SiteTrace(0, "attn"),),
    forward_index=0, sequence_start=0, generation_step=None,
    max_events=4, max_trace_bytes=4096,
)
limits = TraceLimits(4, 8, 2, 4, 16, 16384, 8)
plan = bind_trace_plan(identity, spec, limits, batch_size=1,
                       sequence_length=3, layer_count=2)
capture_request = plan.capture_request()   # a fresh metadata copy
starts = plan.sequence_starts()            # {"run-unique-to-this-forward": 0}
```

The identity, input dimensions, layer count, limits, and declared device and
transport are assertions supplied by trusted code. The binder does not inspect
the worker, actual inputs, model, device or transport. It accepts only the
declarations `device="cpu"` and `transport="in_process"`; this is not an
execution or IPC guard. It cannot authenticate a client or prove that an
input row belongs to a team. A future broker owner must perform those checks,
hold the model lease, and reject unsupported actual devices and transports.

The broker/job identity uses a 32-character lowercase hex broker ID and a
positive job number. `plan.worker_request_id` and the capture's `job_id` are
`lg-{broker_id}-{job_id}`, so they remain distinct across broker lifetimes.
The capture's `request_id` is the original client request ID; the team ID is
kept in `plan.identity` and must travel in a trusted delivery envelope. This
binding is not authentication, and client IDs must not be substituted for
trusted broker IDs. Avoid putting sensitive input text in these identifiers.

Each sample corresponds to exactly one row in tuple order; sample IDs must be
unique within the job. Token positions are distinct, zero-based positions in
the hook output's local sequence axis, not absolute generation or cache
coordinates. Sites identify GPT-2 block, attention, or MLP layer outputs;
they are distinct and restricted to declared layers. A trusted integration
must verify that the actual model supports those sites and explicitly convert
each `SiteTrace` to the exact `TraceSite` type expected by the traced worker.
The supplied `generation_step` is owner-asserted; generation stage and cache
coordinates remain unknown. No guarantee is made that a selected hook runs:
experiments that need complete coverage must check it independently.

Event identity is `(run_id, sequence)`, not the broker-scoped job ID. Two
independently valid plans can use the same run and overlapping sequence ranges.
The binder does not coordinate plans. Before enabling execution, a trusted
broker must allocate unique run IDs per forward or enforce disjoint sequence
ranges across all producers for each run. A reused run must also have a
consistent forward index for each physical root forward, advanced for distinct
forwards. Neither global uniqueness nor cross-plan forward consistency is
enforced here.

The policy bounds the number of rows, sites, selected tokens, events and
retained trace payload bytes declared for a single job. It checks sample/site
tuple lengths before copying those selections, and validates all copied values
before returning a plan. These checks do not reserve memory or bound caller
input objects, Python metadata, model buffers, allocation temporaries, logits,
host copies, allocator overhead, other requests or GPU memory. A future
integration must conservatively reserve and pin separate claims before each
input, output, trace, workspace, store, or transport allocation; caps must
include the number of independently retained items. Claims must remain charged
until all corresponding aliases and asynchronous operations have actually
ended. Cancellation alone is not disposal. Release requires a trusted
custody/discard and acknowledgement protocol covering success, late cancel,
errors, identity mismatch, disconnect and shutdown; finalizers or timeouts do
not establish that clients disposed their copies.

The plan stores copies of exact primitive values, and ordinary construction
or mutation of `BoundTracePlan` is rejected. Getters reconstruct independent
metadata. The caller must not mutate inputs while binding takes place; copying
several Python fields is not atomic. This is a cooperative in-process contract,
not a sandbox against code that deliberately accesses private fields or uses
`object.__setattr__`. Any eventual execution boundary must revalidate trusted
facts and establish exclusive ownership independently.
