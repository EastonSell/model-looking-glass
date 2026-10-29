# CPU trace IPC, protocol version 2

`TraceIPCServer` exposes the reviewed `CpuSharedTraceService` to two trusted
local client processes. It is a separate loopback endpoint and protocol from
the existing forward-only [IPC v1](IPC.md). It serves one float32 CPU GPT-2 worker
serially; it does not batch requests, provide a fairness guarantee, use a GPU,
or authorize any GPU trace profile. This first slice accepts one input row and
one sample per request, with bounded token positions and trace sites.

The owner creates the CPU worker, host budget, trace store and shared service
as described in [CPU_SHARED_TRACE_SERVICE.md](CPU_SHARED_TRACE_SERVICE.md),
then starts `TraceIPCServer(service, max_event_bytes=..., max_trace_bytes=...)`.
The operator must retain the server, service, owner, model and budgets until
clean shutdown or supervised reconciliation. The server mints distinct team
tokens and publishes a loopback address, broker ID and process-generation ID.
Deliver each token and the fresh endpoint privately to its intended client.
There are at most two live connections; the existing service allows one active
job and serializes delivery before the other team can run.

The client does **not** authenticate the server. As with IPC v1, a stale port
can be reused by another local process and the bearer token could be disclosed.
This protocol assumes trusted cooperating local processes and no hostile
same-account code; team tokens authenticate possession, not a person. It is
not a public, multi-tenant, encrypted, or adversarial isolation boundary.

## Request and event handoff

Each client creates its own independent `HostBudget` and a `TraceIPCClient`
with the privately supplied team token, address, broker and process generation,
and locally configured event and total-byte limits. A client object is
single-use after completion or an incomplete attempt. It submits a request
with `input_ids`, `sample_id`, `positions`, `sites` and a synchronous
`on_event` callback. Request IDs and sample IDs are bounded ASCII identifiers;
clients must not reuse a request ID within the service lifetime. The client
makes capped copies of the input row, positions and each site so later caller
mutations do not change validation, encoding or receipt checks. Those copies are
not an atomic snapshot across lists; callers must avoid changing a request while
submitting it if they need a consistent set of values. The server
validates input IDs against the bound model vocabulary and the exact service
shape, site, event and byte limits before making a tensor or calling the model.
No request can supply code, a model, a Python factory, or a resource profile.

Controls use the existing four-byte big-endian length plus bounded UTF-8 JSON
framing. The configured control-frame ceiling is 8 KiB to 64 KiB; smaller
ceilings are rejected at construction so a bounded request cannot be admitted
under a ceiling too small for its expected event header. Each event has a JSON
metadata header followed by a separately
length-checked raw byte frame. The header binds team, request, broker, process
generation, job, job generation, transfer ID, unique event ID and complete
trace metadata. The receiver checks the exact types and identity, requested
sample, site and positions, sequence, dtype, shape, byte order and byte caps
before allocating the payload. No pickle or executable deserialization is
used. Server and client reserve a fixed control-byte envelope before reading or
encoding a control frame; the receiver reserves its payload claim and roots
the allocated bytearray before the first `recv_into`. The sender writes a
memoryview of the already-claimed serialized event, without making another
bulk payload copy.

For each event the receiver first sends `ready` after its reservation, then
`received` after the complete payload has arrived. It calls `on_event` with
the event metadata and borrowed bytearray, drops its own aliases and releases
the receiver claim, then sends `released`. Every message repeats the exact
event and team identity; malformed or mismatched acknowledgments fail the
delivery. A callback must return `None` synchronously and must not retain the
metadata, payload or aliases. A callback that needs a persistent copy must
reserve a separate claim **before** making that copy. A callback failure or
short read retains the receiver's claim, buffer and failure graph.

An early event can reach the callback even if a later event fails. Stage any
derived observations and treat them as a complete trace only after `trace()`
returns a `TraceIPCCompletion`. After all events have been released, the
server sends `complete`; the client validates the total count and bytes and
acknowledges it, and the server acknowledges the underlying local receipt.
The final `closed`/`closed_received` exchange records the client's observation
of this completion. The client may return after sending its last message while
the server has not yet received it. A local completion therefore does **not**
attest that the server retired its session, input or control claims. Check the
server's own quiescence and `incomplete` state separately.

## Failure, limits and custody

A malformed request or wrong team token can be rejected before a model call.
After admission, a timeout, short frame, callback error, wrong identity,
disconnect or lost acknowledgement is `TraceIPCIncomplete`, not success or
proof of non-execution. Do not automatically retry: a forward or partial
delivery may already have happened. The affected client retains its exact
session, failure graph, and any outstanding receive/control claims. The server
retains its incomplete session and any input or control claims still live;
claims already retired after a verified handoff are not recreated. When a delivery
callback fails, the underlying shared service also quarantines and retains its
separate callback-exposure claim and borrowed bytes. Publication and job-start
decisions made after the server latches an incomplete session are refused.
A decision ordered before that observation can still start or finish its send
(including a completion), or start its job, afterward; it cannot be revoked.

`server.close(timeout=...)` requests that new connections stop and waits for its
handlers; it returns `False` if the lock or a handler is still busy at the
deadline, or if any session is incomplete. If it cannot obtain the lock before
the deadline, the caller must retry the close request.
It does not close the caller-owned service or worker. A supervisor must keep
the exact objects and claims rooted until quiescence and disposal are verified,
or terminate and reap the affected processes. There is no automatic retry,
cross-process recovery, or cancellation API in this slice. The server does not
watch the socket while an accepted job waits behind another team: that job may
still run after its client times out or disconnects, with the loss discovered
when delivery is attempted. Arbitrary repeated
asynchronous interruption and non-cooperating native code remain outside the
tested Python-thread contract.

The server and each client use **independent** host ledgers. The server's live
event can overlap the trace store copy, serialization bytes, shared-service
callback-exposure claim, input claim and control envelope; the client has its
own receive and control claims. The fixed control envelope is 8 × 65,536
logical bytes per connection, even when the configured frame cap is smaller,
to cover bounded overlapping control-byte representations. These cooperative
claims are not a combined process-tree limit or an RSS cap. Model parameters,
workspace beyond its operator-declared profile, Python object overhead,
allocator/library allocations, kernel socket buffers and client-made copies
are not measured or enforced by these claims. A `received` or `released`
message is a cooperative attestation and cannot prove an arbitrary remote
process has disposed every alias.
