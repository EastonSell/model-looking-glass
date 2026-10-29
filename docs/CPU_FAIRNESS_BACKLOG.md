# Synthetic CPU backlog and queue-rotation study

`looking_glass.cpu_fairness_backlog` tests the observed serial service with a
bounded synthetic workload. It runs the real fixed-workspace CPU worker,
broker, authenticated loopback IPC and event recorder; the model, guard,
leases and memory telemetry are synthetic. This is not a GPU, production,
isolated-latency, or resource-profile experiment.

The single frozen plan has three teams and four distinct jobs per team, a
30 ms requested sleep in each synthetic forward, 12 global pending positions,
four pending positions per team, and 12 global connection slots. Python and
the OS may make actual service durations longer; the report retains the
observed CPU forward intervals. All source and event metadata are trusted local
observations, not authenticated evidence of hardware execution.

The plan specifies two distinct windows with the same inputs, seed and model
specification:

* `baseline` sends the jobs globally one at a time in the frozen order
  bulk, peer-b, peer-c, repeated four times. The next client attempt starts
  only after the prior response has been handled. It has no competing
  **experiment clients**, but external host activity is not controlled. It is
  not an isolated-latency result.
* `backlog` starts a separate client per job. Bulk jobs are scheduled at
  0, 3, 6 and 9 ms after the common start stamp; peer-b jobs at 15, 75, 135
  and 195 ms; peer-c jobs at 20, 80, 140 and 200 ms. These are planned release
  offsets, not guarantees about when threads, connections or broker admission
  run. The report keeps both the due time and actual client attempt time.

Each window uses a separate warmed synthetic owner. The labels and input hashes
match, but a shared model instance or unchanged resident state is not claimed.
The two windows have different arrival patterns and client concurrency, so
their rates and waits must not be interpreted as a controlled speedup.

## Freeze and run

Use one fresh process per window, each under an external watchdog. First freeze
the plan and retain its printed canonical digest independently:

```text
python -m looking_glass.cpu_fairness_backlog plan --output plan.json
python -m looking_glass.cpu_fairness_backlog run --plan plan.json --sha256 RETAINED_DIGEST --mode baseline --output baseline.json
python -m looking_glass.cpu_fairness_backlog run --plan plan.json --sha256 RETAINED_DIGEST --mode backlog --output backlog.json
```

All outputs must be new. Plan validation happens before owner startup; an
interrupted run may leave an empty or partial report. Preserve the plan,
digest, raw reports, exact source and runtime, and watchdog observations.
The digest detects changes against the retained value; it does not authenticate
the producer or certify a profile. A Python timeout cannot safely recover a
stranded lock or callback; an independent process watchdog must terminate and
reap an unresponsive owner. An uncertain failure retains the in-process owner
root until process exit and may prevent any final report.

## Boundaries and replay

The window starts just before the common release event and ends after clients,
handlers and broker work have drained. Setup and warmup precede the window;
sealing, summary generation and file output follow it. The common denominator
includes scheduled delays, connection setup, serialization, result validation,
hashing, recording, and IPC/broker shutdown. Server polling and teardown can
dominate a short run. Every team rate uses the same full denominator.

Each attempted job records the actual client attempt, post-decode admission
receipt and post-decode/post-close terminal receipt. The terminal IPC response
also contains the broker's original submitted, ready, started and finished
stamps; these are checked against the separately recorded ready, selection and
terminal events. The forward-entry and CPU-completion events come from the
resident worker's observed boundaries. Entry precedes the final owner guard and
is an invocation-attempt timestamp, not proof of model-body entry. Actual
forward durations include CPU output handling. Client receipt is not a broker
cancellation decision or a frame-arrival timestamp. This study never requests
cancellation; timeout, disconnect and unknown acceptance stay unresolved.

The result includes all planned jobs, attempts, observed acceptances,
rejections, failures, unresolved and unattempted work, plus per-team client,
pre-admission, ready-to-selection and other timing distributions. Planned
counts come from the plan; missing and ambiguous samples stay distinct from
unattempted jobs. p95 is nearest rank and equals the maximum for fewer than 20
samples. The resolution screen flags intervals shorter than 100 times the
clock's nominal resolution; it does not prove physical precision.

The optional queue replay uses the actual source-stamped ready and selection
events, never recorder order or numeric job-ID order. For this exact serial
broker, ready is stamped immediately before `queue.put`, and selection is
stamped after `queue.pop`; both happen under the same broker condition. A
strictly earlier ready stamp therefore precedes the later successful dequeue.
The replay starts from the empty queue, applies per-team FIFO and round-robin
rotation, and records the eligible teams and per-team pending depths at each
selection. It runs only when every planned job succeeded and all observations
are complete; any equal ready/selection timestamps, unknown work, rejection,
failure or cancellation makes replay inconclusive. A mismatch is reported as a
violation in that observed window, not silently repaired.
The realized-condition fields are `null` when replay is inconclusive or reports
a violation: neither occurrence nor absence of the planned condition is then
established. A `false` value means a complete matching replay did not realize it.
The replay consumes the driver's prior complete-event audit; calling it directly
with a true completeness flag is a trusted attestation, not authentication.

The report states the maximum observed bulk pending depth, the number of
selections with multiple eligible teams, and the number with multiple teams
eligible **while at least two bulk jobs are pending**.
`bulk_backlog_and_peer_competition` is true only when the last count is
positive; planned offsets alone do not establish this. The
baseline's no-competing-client flag requires its replayed peak pending depth to
be at most one. These are evidence of the realized conditions, not a general
fairness, starvation-freedom, or equal-time guarantee. The IPC connection cap
is global and applies before authentication, so this does not establish fair
network admission. Round-robin request turns do not imply equal GPU time.

Recording and hashing are bounded but add unmeasured overhead and can affect
contention. Per-window distributions and their sample counts are descriptive;
do not subtract an assumed overhead. `work_drained` covers the joined synthetic
clients/handlers/broker and checked custody slots, not a model/lease-disposal
or production-supervisor attestation. On successful CPU work drain the driver
drops its strong fixture root and Python may collect those objects. The current
observer remains CPU-only and must not authorize a GPU workload.
