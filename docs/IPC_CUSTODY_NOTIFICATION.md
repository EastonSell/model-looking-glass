# Experimental IPC custody notification

Status: **trusted, optional, CPU-tested local integration seam**. This adds no
guardian, fair-admission implementation, trace job, device scheduler, result
custody transfer, or remote authentication. The normal service remains
forward-only. A guardian must independently implement and test the contracts
below before any owner uses this seam for live work.

The server can report that its own named request/response references were
cleared after a connection ends. That event is **not proof** that the worker
finished, that the broker's retained outcome or serializer traceback was
disposed, or that socket/kernel/client buffers or any external alias were
released. The guardian must hold its handle, claims and accounting until its
own separately established completion and disposal conditions are met.

## Experimental owner-completion binding

An authority-bound fixed-workspace owner must pass its exact
`completion_authority` to `IPCServer`. The server accepts an exact direct
`Broker`, or an exact `FairIPCBridge` used as its own custody guardian. In
the fair case, the bridge's pinned target, the target's broker, the broker's
fair handoff, and the authority's bound broker must all agree. Before startup
or admission, that bridge must also have rooted this exact server through its
single-server construction guard. A manually constructed, unrooted fair
server is rejected. An omitted or
mismatched authority is rejected for an already bound backend. Genuinely
unbound legacy backends retain their existing configuration. These are trusted
in-process identity checks, not protection against code that mutates private
objects concurrently.

An observed uncertain IPC custody transition calls that authority's
nonblocking sticky `mark_unsafe()` before setting the IPC quarantine flag and
before the handler sends a rejection or continues cleanup. This includes an
uncertain registration, input construction, submission return, observation,
cancellation, serialization, notification, or handler-bookkeeping failure.
An outcome with an unexpected broker, job, team or request identity is not
disclosed; it marks the authority before the failure response and does not
qualify for a clean handler-close notice.
Before accepting a submission, the server checks the exact returned direct
handle's broker/team/request identity, or the fair proxy against its rooted
connection record. Invalid handles are not used for cancellation. Direct
request IDs can be reused, so matching handle fields alone do not prove that a
handle belongs to this particular call: that still depends on the exact
trusted broker's return contract. A separate pre-registered submission
identity would be needed to independently recover or prove a lost direct
handoff. Cancellation returns must be exact booleans; a malformed return is
uncertain and cannot produce a cancellation acknowledgement.
The exact direct broker's ordinary rejection is accepted as clean only under
its own no-unknown-custody contract; exact guardian capacity denial retains
the no-mutation contract described below. Authority-bound constructor and
listener/handler setup or teardown failures also signal the owner. If accepted
socket or handler startup is uncertain, the server reports incomplete
shutdown; supervised process replacement is needed unless a trusted owner
separately proves a safe recovery. Startup and accept-loop failure tracebacks
are rooted when possible, as is an accepted socket after its explicit orphan
handoff. An interruption after `socket()` or `accept()` returns but before the
returned object is assigned may leave no recoverable Python reference; the
unsafe owner requires process-level recovery.

An ordinary `stop_accepting`, `drain`, or `close` timeout returning `False`
is incomplete and retryable; that result alone does not mark the authority.
The server roots each completed handler in a bounded retiring set before it
leaves the active set. New admissions count both active and still-alive
retiring handlers, and `close`/`drain` join them and prune only after actual
thread exit. A `True` return from `close` or `drain` establishes quiescence
of the server's tracked acceptor and handler threads under this cooperative
Python contract; the owner must still check the sticky authority/quarantine
state after the join and establish broker, worker, device and external-alias
disposal separately. Exceptions escaping the protected handler call are marked
by its wrapper.

For an authority-bound server, each handler and the single acceptor have
separate preallocated fault slots rooted before their threads start. On
supported GIL-enabled CPython 3.10 through
3.13, a per-thread wrapper of CPython's private `_invoke_excepthook` marks the
exact authority unsafe before attempting capture, event signaling, or invoking
the existing application hook. A stalled capture therefore cannot allow a
later owner success decision. The wrapper then roots the exact exception and
its original traceback separately. It takes no server lock. The traceback
comes from the active interpreter exception
context and must be an actual traceback for the same exception; a subclass's
overridden `__traceback__` property is not trusted. `retained_handler_faults`
returns trusted-local handler snapshots;
`capture_complete` stays false until both references are rooted, including
when capture is interrupted. An observer failure is separately retained in
the same slot; under CPython's ordinary exception chaining it also roots the
original target fault if signaling or capture was interrupted. An incomplete
snapshot is still uncertain and requires recovery. The property does not wait for the server lock
and raises `IPCError` if it is unavailable. An in-flight snapshot can be
incomplete; join the handler before interpreting it. Without taking the server
lock, `retained_acceptor_fault` returns the corresponding acceptor snapshot or
`None` if no fault has been observed; join the acceptor before interpreting
it. An observed acceptor fault also makes shutdown incomplete. Do not send these
exceptions or their potentially sensitive frames over IPC.

The existing application hook, unraisable hook and supervisor are trusted to
preserve any traceback frames, exception chains, attributes and payload aliases needed for
custody. Rooting the traceback separately survives clearing the exception's
`__traceback__` attribute, but cannot prevent `traceback.clear_frames` or
other mutation of the frames and aliases. An untrusted or noncompliant hook
invalidates that preservation contract and requires separate accounting and
process-level recovery. `capture_complete` records the initial capture, not
proof that another actor has not mutated the referenced objects afterward.

After a faulted handler exits, the server keeps its fault slot and exception
root and counts that slot against `max_connections`; it does not release the
root on `close` or `drain`. The supervisor must retain the server graph and
perform separate custody reconciliation or process-level recovery. The slot
does not itself identify a connection, job or client: the owner must reconcile
all potentially affected records unless it has a separately trusted binding.
A faulted acceptor retains its separate single bounded slot and exception root
until process replacement; it is not counted as a client connection.
A `True` shutdown return reports thread exit, not exception or payload disposal.
Unbound legacy servers do not install this observer, and accessing their
`retained_handler_faults` or `retained_acceptor_fault` raises `IPCError` rather than reporting an empty
inventory. Authority binding rejects
unsupported interpreters, free-threaded builds and versions; the runtime check
only verifies the expected CPython bootstrap hook name, not protection
against trusted-process mutation of private machinery or runtime GIL changes.

The observer covers handler and acceptor exceptions delivered to that CPython
hook, including exceptions escaping their wrappers' returns. An interruption at hook entry
or in later Python bootstrap, exception reporting, or finalizer code can escape
it. A joined thread and a clear flag therefore do not prove that every possible
thread fault was observed. The owner needs separately reviewed fault
supervision and, when its outcome is uncertain, process-level recovery before
a clean attestation. The live service remains gated until the complete owner
lifecycle is reviewed and composed.
The owner must mark it if its own deadline or custody checks cannot establish
safe teardown. Bound-server startup and handler checks reject an owner they
already observe as unsafe, but do not form an atomic admission barrier against
a racing signal. A broker success decision linearizes at its authority check:
a fault signaled afterward can precede delivery of a success already decided,
and does not retract it. Neither the signal nor the local handler notification
proves that a worker, broker result, socket, serializer, or client alias has
been disposed. The caller must root a partially constructed server and factory;
if construction/return or process state is uncertain, keep the service gated
and use supervised process-level recovery. This primitive does not guarantee
recovery from arbitrary asynchronous Python interruption. A direct sticky
fallback covers a one-shot interruption while requesting the unsafe signal;
it does not establish completion or recovery after repeated interruptions.

The trusted request factory can reject an otherwise valid IPC envelope before
constructing its request by raising the **exact** `IPCInputRejected` from
`looking_glass.ipc`. This asserts that its invocation has created no
request-owned tensors or retained aliases, external side effects, or
asynchronous/native work. IPC cannot verify that assertion: a factory that
raises it after such work violates the custody contract. The signal is
recognized only around the factory call; subclasses and all ordinary factory
errors remain uncertain. If a validating wrapper calls a downstream factory,
it must convert an unexpected `IPCInputRejected` escaping that call into an
ordinary uncertain failure rather than misclassifying it as a clean denial.
The bounded response is `invalid_request` and does not expose exception text.
An already registered guardian still receives its normal handler-close notice
after the handler clears local input references; that notice alone does not
prove release of a guardian claim. This mechanism does not validate token IDs
against a model: a service must independently bind its checks to trusted
checkpoint and workload settings before permitting execution.

## Trusted owner configuration

- Supply `IPCServer(..., team_ids=[...], custody_guardian=guardian,
  custody_epoch=epoch)`. A guardian cannot be used with the shared bearer
  token: team-bound token authentication is required. Token possession proves
  only the configured team label; it does not distinguish principals within a
  team or protect against disclosure of a team's secret.
- `epoch` is a trusted owner's 32-character lowercase hexadecimal lifetime
  identifier. The owner must ensure it is never reused for concurrent or
  restarted servers whose notices could be confused. The server does not
  persist or enforce cross-lifetime uniqueness. Each accepted handler obtains
  `IPCCustodyKey(epoch, connection_generation)` with a strictly increasing,
  positive bounded generation within that server; exhausted generations do
  not wrap. Failed and unauthenticated connections can consume generations.
- The exact broker supplied to the server must expose that same object as
  `broker.custody_guardian` and implement
  `submit_registered(key, team_id, request)`. The server retains that broker
  and guardian pair privately for this mode. The guardian and broker are
  trusted in-process configuration; no wire data selects a callback, key or
  Python implementation. Direct calls to the adapter are not authenticated
  by IPC and must not be treated as such.

## Lifecycle and required guardian behavior

1. Before sending the ready frame or receiving a request, the server calls
   `guardian.reserve(key)`. The guardian must atomically pre-root a bounded,
   discoverable record and reserve any declared read/decode and downstream
   envelopes before the corresponding allocations. A failed or interrupted
   registration may have committed and is treated as uncertain. The guardian
   must keep such records and any claims charged for supervisor recovery. A
   trusted guardian may instead raise the **exact** `IPCCustodyCapacity` type
   from `looking_glass.ipc_custody` only when reservation was denied before
   creating any record, claim, request-owned payload, or pending registration
   for that key, and none can subsequently appear. That is a trusted
   no-mutation contract, not a fact the server can independently prove. The
   server returns `rejected/busy` before ready without notifying the guardian
   or latching quarantine; the connection generation is still consumed. If
   the guardian raises this exception after any such side effect, custody
   guarantees do not hold. Subclasses, other failures, and this exception
   raised at any other lifecycle step remain uncertain and quarantined.
2. Only after validating the per-team token and envelope does the server call
   `guardian.authenticate(key, team_id, request_id)`. It then constructs the
   bounded forward request and calls the paired broker's
   `submit_registered(key, team_id, request)`. Before that method returns, the
   broker/guardian must retain the exact job identity, handle, reservations and
   any eventual result in the already registered record. This requirement also
   covers a lost submit return: a connection key remains known to the guardian
   even if the server never receives the handle. The server cannot enforce the
   broker's internal registration or prove that its result custody is safe.
3. On a normal completion, timeout, disconnect, rejected unauthenticated
   request or other safely unwound path, the handler clears its local `handle`,
   `outcome`, `request` and `message`, closes the socket, and calls
   `guardian.handler_closed(key)` with the identity only. It remains counted
   in server shutdown until that callback returns. The call is a statement
   about these local references and socket close; the handler is still on the
   stack, and it is not a worker completion or disposal acknowledgement.
   A timeout/cancel request does not stop an in-flight worker.
4. An unhandled `BaseException`, uncertain registration/binding/submission,
   failed request construction, failed handle observation or cancellation,
   uncertain result serialization, notification failure, or subsequent
   connection-bookkeeping failure
   suppresses the notice or leaves its outcome uncertain. The nonblocking
   `server.custody_quarantined` property latches an observed uncertain
   transition. If a callback recorded a notice and then raised, the server
   cannot retract it: the guardian must treat notice alone as insufficient to
   release anything and quarantine/reconcile the record. A blocked callback or
   a missing notice can remain pending before this flag is set. A false flag
   is therefore not proof that all records are safe or drained. The server
   latches an observed post-notice bookkeeping failure before its handler
   exits; a supervisor must still monitor the signal and its own records.
   Neither notice nor handler thread exit alone proves payload disposal.
5. The supervisor must monitor the guardian and broker independently, stop
   new admission when custody is uncertain, and retain or quarantine claims
   until it proves completion/disposal or performs appropriate process-level
   recovery. `stop_accepting`, `drain` and `close` retain their existing
   timeout semantics; a blocked callback leaves the handler active and can
   make a bounded shutdown return false. Guardian operations must be bounded,
   nonblocking, idempotent where applicable, and may not retain the handler's
   input/serializer frames. If these assumptions fail, no automatic release is
   justified. A trusted encoder must not retain its argument or frame unless
   separately charged and managed.

The IPC handler does not generally retain a factory or encoder exception and
traceback that it catches for the guardian. The authority-bound thread observer
above retains only an exception that escapes to its CPython thread hook; it
does not recover a swallowed exception. Unreachable synchronous CPU temporaries can be
disposed while their conservative connection claim remains charged; that is
not proof that another alias or operation has finished. Trusted overrides that
start asynchronous or native work, or retain buffers or frames externally,
must establish their own rooted ownership, accounting and completion contract
before starting that work. This notification seam does not provide it.

The quarantine signal and defensive checks are not an atomic service-wide
admission barrier; a concurrent admission can already be in progress as a
different handler latches quarantine. The owner must coordinate shutdown and
reconciliation, including all records that raced the signal. This prototype
does not claim immunity to arbitrary Python asynchronous interruptions or
resource cleanup failures. Lost notification is conservatively pending, not
success, and a peer acknowledgement is not proof that another process freed
its memory.
