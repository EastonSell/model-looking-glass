# Synthetic CPU shared-load experiment

`looking_glass.cpu_shared_load` runs a bounded, forward-only synthetic CPU
experiment through one real `LifetimeWorkspaceWorker`, `Broker`, authenticated
loopback `IPCServer`, and `ServiceSerialObservation`. It is a development tool
for the event and accounting path. It is not an isolated-latency benchmark, a
GPU experiment, a production deployment, a resource-profile certificate, or a
serial-versus-grouped comparison. The model returns its two input values as a
float32 tensor; an optional sleep holds the model call for a chosen duration.
The memory telemetry, leases, and device guard are synthetic CPU fixtures.

## Freeze and run

Use a fresh process under an external watchdog. Freeze the plan and independently
retain the digest printed by the first command before starting the second:

```text
python -m looking_glass.cpu_shared_load plan --teams alpha beta gamma --jobs-per-team 4 --model-hold-us 5000 --start-spacing-us 1000 --output plan.json
python -m looking_glass.cpu_shared_load run --plan plan.json --sha256 RETAINED_DIGEST --output report.json
```

Both output paths must be new. The runner checks the plan digest, exact schema,
and bounded workload before reserving the output path; it never overwrites a
prior plan or report. A failed or interrupted run can leave an empty or partial
report file. Save the plan, retained digest, complete report, source revision,
runtime configuration, and independent watchdog evidence together. The digest
only detects changes against the separately retained value; it does not
authenticate the plan, model, runtime, or event producer. Plan and profile
labels are synthetic identifiers, not approved device profiles.

The plan fixes the ordered teams and jobs, input values and expected output
hashes, one request at a time per team, seed, broker pending limits, synthetic model hold, and
optional per-team start spacing. It caps the workload at 64 jobs and eight
teams. Distinct team credentials are used on loopback. Each team has one
closed-loop client thread; its next request starts only after the preceding
response has been handled. This is not an open-loop or equal-arrival-rate load,
and start spacing makes the offered time differ between teams.

The exact CPU worker is warmed once before broker binding. Setup and warmup are
outside the measured window. All clients then wait on one event. The window's
start stamp is read immediately **before** recording and setting that event;
its denominator includes recorder, release, connection, validation, queueing,
forward, client result checks and hashing, and stopping/draining the IPC server
and broker. The end stamp follows those drains and precedes recording the end
event, sealing, report generation, and disk writing. The separate
`clients_joined_ns` shows when all client threads returned. In particular,
server polling and teardown may dominate these deliberately short windows; the
reported shared-window success rate is not steady-state throughput or isolated
request latency.

## What the report means

The report retains the exact plan, client samples, bounded recorder output,
clock implementation and nominal resolution, source timestamps, and audit
findings. Per-team and total counts include planned, attempted, observed
acceptance, unknown acceptance, success, failure, rejection, cancellation,
unresolved, invalid output, and unattempted jobs. Planned counts come from the
frozen plan, with missing and duplicate/ambiguous samples reported separately;
they are not silently classified as unattempted. Rates for each team and the
total use the same full window denominator. Receipt and handling distributions
are separated by final client state. `client_receipt_ns` is read after the IPC
client has decoded a response and closed its socket; it is not frame-arrival
time. `handled_ns` follows the synthetic output validation and hash. Neither
is the broker's decision timestamp.
Distributions include sample counts, a midpoint median, nearest-rank p95 and
the maximum. With fewer than 20 observations p95 is the maximum; one short,
unisolated window is descriptive and does not estimate a tail or SLA.

The raw events use the field shapes of `BATCHED_EVENTS_V1`. Ready, selection,
forward attempt, synchronized CPU completion, and terminal stamps come from
the bound service observer and its actual native broker/job IDs. Ready is
stamped immediately before serial queue insertion; selection follows dequeue
and its bookkeeping. Forward entry is immediately before the resident model
invocation attempt and the final owner guard, so it does not alone prove model
entry. Forward completion includes the corresponding CPU output handling. The
observer stores metadata and bounded event copies; it does not prove provenance
or a CUDA synchronization. This single-window report is not an analyzer paired
result and does not invent a grouped profile or held-out evidence.
The audit checks each successful output digest against the frozen synthetic
value. Matching digests are consistency checks and do not authenticate the run.

For each selected job, the queue section reports its ready-to-selection time
and the number of other recorded selections whose timestamps lie **strictly**
between its own ready and selection. The count is a lower bound when timestamps
tie; endpoint ties are counted separately. It describes turns among jobs that
actually entered the observed broker queue. The connection limit is global
and applies before authentication, so one team can occupy connection slots and
other teams can be rejected before fair queue admission. The experiment does
not establish end-to-end admission fairness, GPU-time parity, or starvation
freedom. Unattempted jobs and failures remain visible rather than disappearing
from the denominator.

This workload does not request cancellation. A lost response, disconnect, or
server timeout is unresolved; the server may attempt cancellation, but the
client receipt is not an observed broker cancellation decision. Do not infer a
decision timestamp, terminal outcome, or successful completion from a timeout.
The audit rejects missing, duplicate, mismatched, or out-of-window observations;
recorder loss or an unresolved cancellation invalidates queue evidence. Rates
and sample counts remain descriptive when an audit is incomplete and must not be
reported as a valid complete measurement. The runner rejects a clock that is
not reported monotonic and non-adjustable before starting an owner. The
nominal-resolution screen flags
intervals below 100 times the clock's advertised resolution; this is a warning,
not a guarantee of physical clock precision.

`work_drained` means the joined client/handler/broker threads and the checked
work and custody slots were clear while the synthetic owner stayed healthy.
Fixed-worker closure does not explicitly dispose its model or release its fake
leases. After a successful CPU run the driver drops its strong owner root, and
Python may collect those synthetic objects; their disposal is **not** attested.
This is not a production process-quiescence or supervisor attestation.
If a callback, thread, shutdown, or owner is uncertain, the runner raises and
keeps the in-process owner graph strongly rooted; replace the process rather
than starting another window in it. A Python watchdog cannot forcibly interrupt
a stuck recorder or lock, so the caller needs an independent process watchdog.
An unsafe model or worker failure can make shutdown raise before a report is
produced; no final success or failure count is fabricated for that window.

Instrumentation uses a bounded recorder, but it takes locks and copies event
metadata; output validation and hashing also run inside the window. Their
actual overhead and effects on contention and ordering are not established by
this experiment. A later performance comparison must characterize recording and
hashing separately in every mode, use a shared frozen protocol, verify actual
model/input/runtime identity and ownership, and obtain a separately measured
profile for the instrumented service. The current CPU-only observer must not be
used to authorize a GPU path.
