# Bounded concurrent clients for the direct owner

`ConcurrentClients` submits one fixed request for each of the two configured
teams over separate authenticated loopback connections. The existing direct
owner executes them on one serial broker and one model. Each request uses the
profiled input `[[15496, 995]]` and one of the fixed seeds 29, 43, or 71. This
is a one-shot, two-request client composition, not a new owner or a general
request router. It shares the prepared service's one-shot claim with the
direct smoke. The caller must provide exclusive use of that controller while
the clients are active.

The direct-owner and completion-custody compatibility markers are still
absent. This interface cannot enable the GPU service; the prepared service
also checks its exact source, runtime, collector, approved load/cold/warm
profile IDs, physical lease and supervisor before starting the child. A new
reviewed, source-specific custody profile and fresh coordinated device
admission are required before GPU use. A profile from the grouped pilot or a
different model path does not authorize this composition.

The caller retains the prepared service and the `ConcurrentClients` object,
starts the service using its existing API, then starts the client object once.
`wait(timeout=...)` returns `None` while client startup, a client thread, or a
cancellation send remains active and may be called again. Once a report is final, a later
`cancel(team)` call is rejected without sending. A rejection frame is not a
clean-custody receipt: the server can emit `invalid_request` after a factory
fault has already quarantined the owner. Every rejection therefore requires
reconciliation, as do send failures, timeouts, missing terminal responses,
worker failure, OOM and uncertain client threads. Do not retry as a new request.
Retain both objects and reconcile through
the existing service shutdown, supervisor result and unsafe marker. A false
`requires_owner_reconciliation` means only that this adapter observed none of
its listed client faults; it is not proof of successful client-thread or owner
teardown and never replaces the supervisor and marker checks. Even a report
containing two successful responses is not a teardown attestation.

Cancellation can be requested at admission by the constructor or later by
`cancel(team)` after `wait_admitted(team)` reports that admission. A false
admission wait can mean a timeout and does not prove rejection. A successful
cancel send is not a broker acceptance. `cancel_ack` records the validated acknowledgement
only if one is received; `None` means it was not observed. A running forward
may finish before cancellation is decided. Cancellation does not preempt a
native operation, retract a result whose publication already won, or override
an OOM or owner failure. A cancellation send error requires reconciliation.

The report contains only two bounded request observations, normalized errors,
broker timestamps, job identities and a SHA-256 digest of validated float32
logits. It contains no endpoint, credential, raw logits, activation trace or
physical device identity. Source timestamps can show that the two observed
broker execution intervals did not overlap and that one accepted request was
ready before the other finished. They are not GPU kernel timestamps. The
two-connection limit does not establish fairness under sustained competing
backlog, throughput, or simultaneous device execution. The report explicitly
sets `fairness_under_backlog` to `not_measured` and `activation_trace` to
`disabled`.

The owner caps each request frame at 4 KiB, each response frame at 4 MiB, and
connections at two. Those wire limits do **not** bound Python object, tensor,
JSON conversion, or total process RSS. The worker's device profile does not
measure the concurrent IPC host envelope. A reviewed host reserve and measured
source-specific device/host limits are separate admission requirements. The
client's bounded thread-fault observation uses the same supported CPython
thread-hook mechanism as the IPC server. A lost submit return, interrupted
hook/bootstrap/finalizer, unresponsive native code, or repeated asynchronous
interruptions can leave an unknown outcome; no client-side report establishes
that the owner has stopped in those cases. The supervisor remains responsible
for process-level recovery.

A trace-enabled successor needs a separately versioned execution identity,
an explicit bounded capture and delivery plan, GPU and host trace reservations,
failure/alias custody and independent exact-source measurement. The current
unobserved profile and its disabled recorder policy cannot certify such a path.
