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

The IPC handler does not retain the original factory or encoder exception and
traceback for the guardian. Unreachable synchronous CPU temporaries can be
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
