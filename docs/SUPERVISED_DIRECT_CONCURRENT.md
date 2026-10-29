# Supervised direct two-client composition

`SupervisedConcurrentRun` composes the existing prepared fixed service and its
one-shot `ConcurrentClients` plan. Construct and retain the prepared service,
client plan, and run object before calling `run()`. The composition starts the
existing supervised owner, starts the two clients, waits for their bounded
metadata, and then requests shutdown through the same prepared controller. It
publishes a result only after a clean, reaped, zero-exit supervisor result, an
absent unsafe marker, and a repeated source/asset preflight. The result includes
the source-base and asset-manifest SHA-256 values and the three approved worker
profile IDs. The client report keeps its own cancellation, failure, queue and
no-trace semantics.

This entry is currently **disabled**. The direct-owner and completion-custody
markers are absent, and `_require_measured_host_envelope` deliberately rejects
every run before starting the owner. The fixed worker's GPU profiles do not
measure the two-client IPC codec, Python object, serializer or aggregate host
RSS envelope. The per-frame caps are not a host-memory guarantee. Before this
entry can launch, the exact two-connection owner/codec/output path needs a
separately reviewed source-bound host observation and admission contract. A
bounded development-only smoke may collect observations under its own reviewed
resource and recovery rules; an observation is not a worst-case RSS bound.
This gate protects this entry only. The direct client and prepared service APIs
must also remain unavailable for live GPU use until their shared admission
requirements are met; this module does not add a global host-memory guard.

The existing fixed owner independently reconstructs its physical identity and
the custody-bound model-load, cold and warm profiles under its device lease.
The grouped 1×4/2×4 pilot and unrelated one-shot GPU results do not authorize
this serial 1×2 path. After the owner, concurrent-host admission and all
compatibility markers are finalized, freeze the final source and runtime,
collect and independently approve matching custody profiles, and perform fresh
under-lock device and host headroom checks in a coordinated slot. Source changes
after measurement require new matching profiles. This composition enables no
activation recording; a trace-enabled successor requires its own versioned
path, budgets and measurement.

The prepared service and clients must have exclusive cooperative ownership
while `run()` is active. A caller may use the client's existing admission and
cooperative cancellation APIs from another thread. A client timeout, startup
interruption, failed shutdown, lost response, rejected request, OOM, or missing
supervisor result can leave work uncertain. The run object retains exceptions
from its run, shutdown, supervisor-wait and source-check boundaries, but the
operator must retain all three objects and reconcile the original controller,
client threads and unsafe marker. `wait_supervisor()` rechecks the pinned
controller even if the prepared binding changed. Do not replay an uncertain request. The
existing supervisor's process-level watchdog remains necessary for stalled or
unobservable thread/native failures.

The checked source inventory and profile IDs are trusted reproducibility
bindings, not a sandbox or proof of every loaded Python/native byte. A clean
run also checks the canonical prepared-service and client methods before and
after startup; this is a point-in-time cooperative integrity check, not an
atomic defense against arbitrary in-process mutation. A clean
supervisor result and an absent marker are scoped to the supervisor's documented
protocol and the instant of observation. A client report with no detected fault
is never itself evidence of teardown. The returned metadata contains no raw
logits, credentials, endpoint, local paths or physical device identifier, and
does not establish sustained-backlog fairness or simultaneous GPU execution.
