# Experimental IPC fair-broker bridge

This owner-private bridge has CPU tests, including real loopback sockets and a
CPU worker. It provides a narrow forward-only path from the existing IPC
server through a fair waiting adapter into one resident broker. It does not
enable batching or prove concurrent GPU execution. The bridge does not verify
the worker device, checkpoint, GPU admission, runtime identity, model behavior,
or a resource profile. A GPU service needs its own reviewed composition,
whole-workload memory admission, device lease and supervisor before use.

`FairIPCBridge` requires a `ForwardCodec`, a bound `ExperimentalFairAdapter`,
one distinct, owner-held adapter credential per team, a unique custody epoch,
and explicit connection limits. The broker fair target must have been built
with that exact codec's `decode` method and item/decoded-tensor limits at
least as large as the codec limits. Construct the server with
`bridge.make_server(...)`; it fixes the paired server's team-token mode,
connection count, request/response caps and input shape limits. Direct calls to
guardian and adapter methods are trusted internal APIs, not authenticated
network endpoints. IPC team-token verification authenticates only the configured
team label to the owner-private guardian. The bridge maps that label to the
corresponding private adapter credential, and the adapter verifies that the
credential's principal belongs to that exact team. Sharing or disclosing team
tokens undermines this identity boundary; loopback itself is not server
authentication against hostile local processes.

The codec accepts exact CPU int64, detached, contiguous rank-two forward
tensors with the IPC input names and compatible shapes. It checks the aggregate
element limit before conversion to Python lists, and checks aggregate decoded
tensor bytes before constructing them. Source tensors must have compact,
zero-offset PyTorch-visible storage. A trusted factory must bound backing
storage before returning: this check cannot undo an earlier allocation, and
it cannot prove how much memory an external NumPy/buffer owner or allocator
retains. The codec's byte limit is a logical
tensor/storage bound, not a bound on Python objects, allocator overhead or
model intermediates. It takes a private CPU snapshot before invoking the
optional trusted `request_validator`; the validator must enforce model-specific
token/position/seed/profile constraints without mutating or retaining the
snapshot. A `ValueError` is an ordinary per-request rejection only when that
trusted validator has cleaned up its own temporary state. Without such a
validator, the generic paired builder does not establish model-domain safety
and must not be exposed to live untrusted requests. `make_server` also accepts
a trusted `request_factory` for the service's source validation and private
input creation; it must obey the same custody and ownership contract. No caller
may mutate inputs concurrently while the snapshot is being made. The codec
does not accept trace or intervention requests.

Before the server reads a connection, the guardian roots a bounded connection
record and reserves a separate `HostBudget` claim for input, recipient and
serializer custody. An owner-specified `connection_commitment_bytes` must
cover all simultaneously live request/frame/JSON/tensor and result/serializer
objects, including integer or NumPy intermediates, lists, JSON strings, socket
framing, aliases and retained tracebacks. The constructor requires a conservative
floor based on configured byte limits, but that heuristic is **not measured or
proved** as a physical peak. The owner must profile the actual Python, encoder
and worker path, select an adequate commitment and keep global process/device
limits separately. These claims are cooperative declarations, not OS-enforced
memory limits. Client/kernel buffers, other processes and arbitrary model
workspace require separate accounting.

The guardian records the authenticated team and request ID before accepting
the exact `ForwardRequest`; it retains the request and a recoverable
registration guard before queue publication. The bridge assigns its own
lifetime broker and job IDs and maps a target result back to those IDs only
after checking the target's broker, team and request identity and the exact
live, published, unacknowledged ticket-to-handle receipt under the broker's
condition. This depends on the bridge being the exclusive receipt/ACK owner;
the bridge condition is acquired before the broker condition. It rechecks
custody before delivery and observes broker quarantine independently. A single
trusted owner that loses a successful submission return can use
`recover_proxy(key, team_id, request_id)` to obtain the same rooted proxy;
these fields are custody identity checks, not authentication credentials.
After an interrupted cancellation return, the retained proxy's
`cancel_requested` reports recorded intent, not proof that execution stopped.
These recovery methods do not make an indeterminate transition safe to reuse.
Each handler-owned proxy allows one operation at a time. Concurrent wait,
cancel, or cancellation-state calls receive `ProxyBusy` without changing the
request; retry after the current call finishes. Use short wait polls when a
cancellation may be needed. The paired IPC handler is the sole proxy user;
no separate owner may access that proxy while its handler is active. A direct
trusted owner must finish all proxy operations and drop its aliases before
notifying `handler_closed`. Different connections share the bridge condition:
preparation, validation, encoding and other critical sections can delay peers.
Trusted validators must be bounded and nonblocking; no latency fairness is
promised. Internal uncertain transition-owner, gate and wait errors have
separate bounded private roots.
Exactly one service-owned thread may call `transfer_next`; no other dispatcher
may use the adapter or acknowledge receipts concurrently. Its round robin covers
admitted eligible team waiters; global byte saturation, pre-auth connection
slots, a blocked decoder, work duration, device time and completion latency
are not fairly scheduled by this component. A known waiter-capacity rejection
is reported as a terminal `QueueFull` result for that connection without
quarantining other jobs. Admission is still bounded independently by guardian
connection records, waiting bytes, broker slots and target commitments.
If detached work retains all connection commitments, a later connection is
rejected as busy before a new guardian record or claim is created; this known
capacity denial does not itself quarantine the service. An ambiguous failure
after registration begins is not a capacity denial.

Cancellation of a waiting ticket removes that ticket for its verified owner.
Once claimed, cancellation is cooperative. During decoding a request can be
recorded before a target handle exists; it is forwarded when the handle is
published, and may lose to completion. The bridge then reports the actual
target outcome, including success or failure. A cancellation request does not
guarantee preemption, output suppression, or immediate memory release. A
finished target outcome is never rewritten by a later cancellation.

`handler_closed` records only the IPC handler's reference-clear/close notice.
For an actual paired server, the bridge waits for that exact handler thread to
exit and checks the server's nonblocking custody-quarantine signal, then waits
for target retirement and acknowledges the exact receipt. Only after those
steps and its own alias disposal does it release the connection claim. It
stages owned aliases under the bridge condition and drops them outside that
condition, keeping the claim charged until disposal returns. A stuck finalizer
can still stall its one dispatcher and requires supervisor recovery. This
depends on the IPC server latching **all** uncertain cleanup failures before
handler exit; the bridge must not be deployed against a server version that
does not provide that property. A notice is not proof that client/kernel
buffers or other external aliases were disposed. If registration, notification,
transfer, target health, result serialization or cleanup is uncertain, retain
claims and stop admission for supervisor-led process recovery. Quarantine flags
are not an atomic global admission barrier, and arbitrary Python asynchronous
interruptions can strand locks; use an external watchdog. The supervisor must
monitor the server, broker, worker and bridge independently.
An in-process service owner must also keep a strong reference to the entire
bridge/server/guardian object graph until every claim is cleanly released or
the owner process is replaced; an external watchdog cannot root Python objects.
Bridge-owned ambiguous failures retain their private exception/traceback and
return sanitized public errors. Synchronous CPU IPC factory and serializer
temporaries can be disposed on normal stack unwinding while the declared claim
stays charged. A factory or serializer that launches asynchronous/native work
needs an independently rooted operation and custody contract; this bridge does
not establish that contract.
The decoder keeps successfully constructed tensors in a named local so a
retained fatal/OOM traceback can root them. This relies on the separately
reviewed fair target retaining its private fatal decoder exception. Arbitrary
interruption between a tensor allocation and its Python assignment remains a
trusted producer/runtime boundary, not a demonstrated general guarantee.

For a clean shutdown, stop and close the paired server first, then close the
bridge and the broker. `bridge.close(timeout=...)` returns false if records
are still live, the dispatcher is blocked, or custody is quarantined; it does
not silently release uncertain claims. Timestamp values use
`perf_counter_ns`; the bridge submission time begins after IPC decoding and
wire validation. It includes the bridge's private snapshot, model-domain
validation, encoding and fair waiting, but excludes connection/read and IPC
wire validation time. Target ready, start and finish times retain the broker's
same-clock measurements.
