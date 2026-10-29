# Grouped-broker event observations

`GroupedObservation` is an optional, trusted, in-process observation path for
the experimental `GroupedBroker`. It records ordinary metadata for the schema
in [BATCHED_EVENTS_V1.md](BATCHED_EVENTS_V1.md); it does not schedule work,
certify a model or profile, measure GPU utilization, or provide memory
admission. The hooks have been exercised using the actual grouped worker and a
CPU model. They do not establish that a GPU batch has been run.

The owner constructs a `BatchedEventRecorder`, wraps it in
`GroupedObservation(recorder)`, and passes that exact observation as the
`observation` argument to the grouped broker. One observation belongs to one
broker owner epoch and one recording. The trusted driver may call
`observation.record(event)` for observations made at its own boundaries. It
must retain the observation, recorder and broker while producers are alive.
After stopping and joining **every** submitter, cancellation thread, dispatcher
and other event producer, the driver calls
`observation.seal(producers_stopped=True)` and retains the returned raw report.
The `True` argument is an owner attestation, not a join or a device drain.
The report combines the recorder's loss state with a separate sticky hook-fault
flag; do not substitute a report read directly from the underlying recorder.

The driver captures `window_start`, `attempt`, client-visible `returned`,
`owner_drain` and `window_end` at their real sources. It must independently
verify request and output fingerprints, model/runtime identity, the actual
workload-profile digest, held-out numerical evidence, drain and owner health.
All sources use the same process's `perf_counter_ns` clock. Only one input row
per request is supported by this event schema. For an observed run, the driver
must use a globally unique, at-most-128-character `ForwardRequest.request_id`
as the frozen logical job ID. The broker still assigns its own distinct job
ID. Duplicate logical IDs or an unsupported launch invalidate the observation
without changing the broker's normal request semantics. A selected workload's
actual `WorkloadProfile.profile_id` must be a lowercase 64-character hex
digest; the owner must separately establish that it is the digest of the
approved live profile, since its spelling alone does not prove that binding.

The broker stamps readiness when it publishes the validated input snapshot to
the ready queue and selection when its fair queue returns the selected cohort.
It stamps a terminal outcome immediately before publishing the single-job or
atomic cohort result. An unserviceable head and prelaunch cancellations have no
model-call event. For a launched cohort, the worker validates the exact live
launch capability and binds its ordered job IDs to the actually selected
whole-batch workload. At the resident worker, the entry timestamp is read
immediately before the Python call to the model, after device input preparation.
The ordered members, row offsets, actual input shape and profile accompany that
observation. The timestamp identifies a Python invocation boundary; it does not
prove a CUDA kernel began. If an interruption makes invocation uncertain, the
record is incomplete or the call is marked uncertain.

For CPU, the completion timestamp follows synchronous model execution and the
worker's output extraction, validation and CPU copy. For CUDA, it follows the
actual successful device synchronization, which occurs after that output work.
Consequently the interval includes the worker's output work and observation
overhead; it is not a pure kernel duration. A failed synchronization, an
interruption, or an exception before the model returned is reported as an
uncertain call, not an invented synchronized finish. An out-of-memory failure
still fails the entire cohort and preserves the broker's quarantine and
per-member cancellation precedence. No tensor, model output or device lease is
stored in an event.

Recording makes bounded in-memory metadata copies and performs no file I/O.
Event-copy or capacity failures set an independent sticky incomplete flag and
do not replace an ordinary worker outcome; control-flow interruptions are
latched and follow the existing worker/broker failure and quarantine paths.
The recorder uses a lock: it is not guaranteed nonblocking, and a damaged or
stranded producer must be detected by an external bounded supervisor. If any
producer, hook, owner or device state is uncertain, the driver must reject the
run even if an earlier report claimed completeness. Recorder storage and Python
metadata overhead need their own host budget; the worker's tensor claims are
not a bound on them. A report cannot be revoked if the caller falsely attests
that producers stopped before a later write.

There is not yet a matching serial-broker observation path. A grouped-only
record is useful for correctness integration, not an unbiased serial-versus-
grouped throughput comparison. Before making that comparison, instrument both
modes at matching model-entry, synchronized-finish, client and output-hashing
boundaries, use the same clock and recorder policy, disclose overhead, and run
a controlled supervised experiment. Keep its raw observations and separately
record warmup, resource profiles, numerical agreement and device telemetry.
