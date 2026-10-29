# Serial-broker event observations

`SerialObservation` and `SerialObservedWorker` provide an experimental,
trusted in-process serial path for the event schema in
[BATCHED_EVENTS_V1.md](BATCHED_EVENTS_V1.md). CPU tests exercise a real
`Broker`, `BudgetedWorker` and `ResidentWorker`. These hooks do not certify a
model or profile, authorize device memory, prove a CUDA call, or provide an
end-to-end service benchmark.

The single trusted owner creates a `BatchedEventRecorder`, a
`SerialObservation(recorder)`, an exact `BudgetedWorker`, and
`SerialObservedWorker(worker, observation)`. Construct the exact `Broker` with
that wrapper and call `wrapper.bind_broker(broker)` before exposing it or
submitting any work. The owner retains this object graph through shutdown and
uses `observation.submit(team_id, request)` and
`observation.wait(handle, timeout)` for every accepted job. Direct submissions,
an unjoined producer, a lost return, or missing terminal capture cannot support
a complete measurement. Window-end and seal checks reconcile registered IDs
with the broker's monotonic allocated-ID history, including direct jobs that
were cancelled before execution. A failed preparation can consume an ID and
therefore invalidates the report conservatively. This check does not replace
exclusive owner control or supervision of all producers. Request IDs must be
unique logical job IDs of at most 128 characters; the recorder retains the
broker's distinct native integer job
ID. The bounded registry supports at most 4,096 jobs per owner epoch. A direct
broker or fair-IPC submission path is not automatically observed.
The fixed service's `LifetimeWorkspaceWorker` uses a separate
[CPU-only observation adapter](SERVICE_SERIAL_OBSERVATION.md). Its CUDA path
remains recorder-disabled until a separately reviewed instrumented profile is
available.

At the selected `BudgetedWorker` admission, the scope binds the actual
single-row `WorkloadProfile`, resident worker, internal worker request token,
input shape, dtype and device. The profile ID must be a lowercase SHA-256
string; the service must independently establish that it identifies the
approved live profile. The scope does **not** hash or attest admitted tensor
contents. The driver must independently bind the submitted and admitted input
values to the frozen workload with appropriately bounded, symmetric validation
and hashing. It must independently establish unchanged owner, model, checkpoint,
runtime, state and device across the paired windows. An owner-session string or
matching metadata alone does not establish continuity.

Serial and grouped forwards use the same resident hooks: entry is the Python
invocation-attempt boundary immediately before the actual model call, and
finish follows CPU output work or successful CUDA synchronization. That
interval includes output extraction, validation and CPU transfer, not only
model or kernel time. A pre-call interruption is not proof that a kernel ran;
uncertain work invalidates the evidence or is reported as uncertain. The
actual single-row token names the call and its only member has row offset zero.
The CPU paired test uses two separate model instances and a synthetic shared
owner label: it checks schema and output consistency, not model continuity or
comparative performance.

`Outcome` already stores source-stamped `perf_counter_ns` readiness, selection
and terminal times. `observation.wait` checks the exact registered handle and
the broker's exact outcome, then copies those original stamps into the recorder
before returning. A cancelled pending job has no selection or model call. A
timeout or missing/ambiguous observation latches incomplete; retrying a wait
can still return the real result but cannot repair the evidence. Concurrent
waits on the same handle while its metadata is being copied also invalidate
the recording. Copying must finish before the driver records `window_end`; an
outstanding submission or capture at that boundary invalidates the report.
The driver must capture attempts, returned outcomes, cancellation decisions,
drain and window boundaries at their actual sources, and retain the raw events.

The queue timestamps do **not** have identical boundaries in the two modes:
serial readiness is stamped immediately before queue insertion, while grouped
readiness is stamped immediately after insertion. Serial selection follows
dequeue and broker bookkeeping; grouped selection follows the compatible
cohort pop. Do not interpret the resulting queue-delay intervals as identical
without accounting for those differences.

Ordinary observation or recorder failures latch incomplete without replacing a
real request outcome where the original execution remains safe. Control-flow
interruptions propagate through the existing broker and worker failure paths.
The recorder and observation registry use locks and are not guaranteed
nonblocking. An external bounded supervisor must reject a stranded or failed
producer and independently verify owner and device drain. After stopping and
joining every producer, the driver calls
`observation.seal(producers_stopped=True)`; this is an attestation, not a join.
The report must be rejected if any owner, hook, cancellation or device state is
uncertain, even if an earlier report said it was complete. Recorder metadata
requires a separate host-memory budget.

Both modes incur instrumentation work. Serial broker events are copied later
at wait, whereas grouped publication records them at their source; the two
paths can perturb contention and grouping differently. The serial binding
also adds a thread-local lookup to unobserved `BudgetedWorker` forwards. A
controlled paired experiment must use the same frozen source and output-hash
policy in both modes, characterize observer-on versus minimally observed costs
for each mode, retain raw samples, and disclose the queue boundary difference.
No GPU throughput or fairness result follows from these CPU tests.
