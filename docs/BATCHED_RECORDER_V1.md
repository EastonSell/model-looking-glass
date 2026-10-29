# Bounded observations for a batched window

`looking_glass.batched_recorder.BatchedEventRecorder` is an in-memory sink for
the exact event records in [BATCHED_EVENTS_V1.md](BATCHED_EVENTS_V1.md). It
does not install broker hooks, read a clock, run a model, establish CUDA
synchronization, or verify model, input, profile or owner identity. A future
driver must capture each `time_ns` at the actual source using the same
`perf_counter_ns` clock and provide the resulting record to `record(event)`.
Delayed publication may change event list order; the original source time is
retained, and the offline adapter checks the event relationships.

The recorder copies exact built-in JSON primitives under a lock. It accepts at
most 4,096 events and a conservative one-MiB bound on the compact UTF-8 JSON
`{"events": [...]}` envelope; smaller limits can be specified at creation.
Each event's exact fields must match the adapter, strings are bounded, integers
are nonnegative and below 2^63, nesting and list sizes are capped, and inputs
with user-defined types or invalid Unicode are rejected. The bound deliberately
overestimates escaping, so valid data can be rejected before reaching the
nominal limit. It is a bound on serialized metadata, not process RSS, source
objects, or other output tensors. A driver must reserve aggregate host memory
separately. During `record`, the producer must exclusively own its dictionary
and nested storage; a later change to the source object does not change the
accepted snapshot.

These are structural and primitive-type checks, not complete field semantics:
for example, a numeric `job_id` is a supported JSON primitive but is not a
valid job identifier. A successful `record` and `events_complete=true` do not
replace the adapter's later identifier, ordering and workload checks.

`record` returns true only if that event was stored. A structurally invalid event,
capacity overflow or internal copying failure returns false and permanently
marks the recording incomplete; accepted earlier records remain available for
diagnosis. There is no file I/O in `record`. A producer that cannot form or
submit an expected observation must call `mark_incomplete()`. Process-level
interruptions are not swallowed: an interruption during the protected copy
normally marks the buffer incomplete and is re-raised. This is not a
nonblocking recorder: `record` and `mark_incomplete` share a lock, and a
stranded lock or an interruption before the fault latch runs can prevent loss
reporting. The controller must independently detect producer or hook failure,
use an external watchdog and reject a run if producer shutdown or reporting is
uncertain. It must also handle interruption, quarantine and evidence
preservation. Event loss before a hook invokes the recorder is not automatically
detectable; the driver must treat hook installation, failures and shutdown as
part of its trusted producer contract.

After **all** event producers have actually stopped and been joined, call
`seal(producers_stopped=True)` and then `report()`. The argument is a caller
attestation, not proof of quiescence. The detached report contains raw events,
the sticky `events_complete` flag, event count, conservative byte bound and
loss counters. A call after sealing is dropped and makes *subsequent* reports
incomplete. A report already copied or published cannot be retroactively
revoked, so sealing while producers can still run is invalid. Persist the raw
report outside the measured window and pass its completeness flag and events
to `assemble_window`; the adapter still must reject missing, contradictory or
ill-ordered evidence. A true flag means this recorder observed no storage
failure, not that the upstream hooks observed every required event.

The lock and bounded copying consume CPU time in the measured execution path;
any memory allocation, hashing and synchronization in the hooks has its own
cost. Record observations symmetrically for serial and grouped windows,
report the recorder's CPU-only microbenchmark separately, and measure
end-to-end instrumentation overhead in a controlled paired workload before
interpreting it. The microbenchmark does not establish GPU latency or the
actual cost under concurrent broker traffic. Event timestamps, synchronized
forward completion, owner draining, resource admission and raw-output
retention remain independent responsibilities of the trusted driver.
