# CPU observations for the fixed-workspace service path

`ServiceSerialObservation` is a trusted, experimental in-process adapter for
the exact `LifetimeWorkspaceWorker`, its `OwnerCompletionAuthority`, and the
normal broker and authenticated direct IPC server. The CPU-only observation
policy must be explicitly selected at worker construction and bound to the
exact recorder before any broker submission or listener startup. Ordinary
service construction retains its separate, recorder-disabled policy. A CUDA
worker cannot select or attach this CPU test policy.

The service owner constructs a `BatchedEventRecorder` and
`ServiceSerialObservation`, passes that exact observation to the CPU test
worker, and calls `observation.bind_service(worker, broker, server)` after
warmup and before the listener starts. The owner must retain the complete
worker, broker, server, observation and recorder graph until all producers and
their result aliases are drained. Direct in-process tests can bind only the
broker with `bind_broker`. The service binding requires per-team IPC tokens;
it does not authenticate the server to a remote peer. The existing trusted
local transport, input validation, cancellation and result custody contracts
continue to apply. No proxy worker or replacement request handle is installed.

When its callback succeeds, the broker registers the exact accepted handle
before returning it; a skipped or failed registration invalidates the report.
The real broker owner starts the existing serial forward scope, and the lifetime worker
binds it to its actual admitted snapshot and warm `WorkloadProfile`. The model
entry and completion are observed by the resident worker. When terminal capture
succeeds, it copies the broker's original ready, selection and terminal
timestamps after releasing the broker condition and before that caller
receives the outcome. A contending waiter may receive its committed outcome
without capture, which invalidates the report. A server handler's
`RequestHandle.wait(0)` timeout is a nonterminal
poll, not a lost event. An `IPCClient.wait(timeout)` timeout is a different
boundary: the client closes its socket and the server may request cancellation,
so the driver must retain it as an unresolved client outcome.
Missing terminal captures, incomplete registration, concurrent ambiguous
capture, recorder failures or copying that races `window_end` invalidate the
recording. Control-flow interruptions propagate through the existing owner
quarantine paths. Normal recorder failures preserve the real result.
An accurately recorded `forward_uncertain` can have complete event capture,
but is not a synchronized successful model call or valid completed timing.
Terminal and registration metadata callbacks each use one pre-rooted,
nonblocking gate. If a callback is already running, another caller still gets
the real result or handle and the evidence becomes incomplete. The active
callback's owners and failure remain rooted after an interruption, with a
separate preallocated root for a failure while notifying quarantine; entry
failures before callback execution are CPU-only metadata failures and do not
start asynchronous work. Those competing callers can already hold an accepted
handle or a completed CPU outcome; the gate does not transfer that custody.
Their otherwise unreferenced CPU aliases may be reclaimed, which is not proof
that unrelated work or host claims have retired. The owner and IPC guardian
must keep their normal claims charged until independently reconciled. Unexpected
loss of gate or producer liveness requires the external supervisor to reject
the report and replace the owner.

The current IPC client does not expose a source timestamp for the broker's
cancellation decision. A cancelled IPC outcome therefore cannot alone form a
complete V1 cancellation record; a future driver needs a separately reviewed
source observation instead of inferring the decision time from client return.

The final owner/session check intentionally runs *after* the resident entry
observation and immediately before model invocation, so an unsafe signal during
the hook can still suppress the call. Consequently the entry-to-finish interval
includes that final guard and its CPU work; it is not identical to the current
unguarded grouped-worker boundary or a pure model/kernel duration. A future
paired comparison needs an equivalently guarded grouped owner, the same clock
domain and symmetric output hashing, plus measured observer overhead. Clock
comparability across service and client processes must be independently
verified rather than assumed.
The broker now checks for this optional hook even when observation is disabled;
use the same frozen source for compared windows and characterize enabled and
disabled overhead rather than assuming the prior uninstrumented timing.

Profile IDs, request values, checkpoint, source, runtime, model and session
continuity require independent verification by the owner and measurement
driver. Hook metadata does not hash admitted tensor contents or prove device
execution. The fixed service's existing CUDA profile describes recorder-disabled
execution and cannot authorize this path; a GPU-capable policy would require
new safety review and a separately collected, approved instrumented profile.
The CPU tests and loopback sockets do not provide a GPU or throughput result.

Before sealing, the driver must stop and join every client, handler and broker
producer and independently establish owner health and drain. The recorder and
registry can block; a bounded external supervisor must reject stalled or
uncertain producers. The driver must separately budget event metadata and
export its raw report through an approved bounded channel after drain. This
adapter does not add an export channel to the supervised service process.
