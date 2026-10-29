# Experimental fair-admission adapter

`looking_glass.fair_adapter.ExperimentalFairAdapter` is an experimental,
in-process waiting layer. It is **not connected to the current Broker or IPC
server**, and it does not run models or manage GPU memory. The current Broker
lacks the required atomic handoff interface and is rejected as a target. The
example target in tests is a CPU-only simulation, not a broker implementation.

The adapter combines the protected team waiter positions described in
[FAIR_ADMISSION.md](FAIR_ADMISSION.md) with a trusted identity verifier,
owner-scoped cancellation, immutable payloads, and byte limits. It exposes no
network endpoint. Only `bytes` payloads are accepted; mutable arrays, tensors,
and bytearrays are rejected. A production integration would need a bounded,
validated decoder and a separately budgeted snapshot before model execution.

The verifier is injected by the trusted owner and must authenticate the supplied
credential before returning an exact configured principal ID. The adapter checks
that the returned ID is a configured string; it cannot prove that the verifier
authenticates, that credentials have been distributed securely, or that the
principal mapping assigns aliases to the correct canonical team. Never pass a
caller-supplied identity through an identity function and call it authenticated.
The adapter and its target must remain inaccessible to untrusted clients except
through an authenticated, authorized service. Direct access to a broker's
submission API bypasses this admission policy.

`transfer_next()` and `close()` are trusted service-internal operations. Neither
authenticates a caller; their receipts and canceled-ticket lists can refer to
other principals, so they must never be exposed as untrusted client methods.

`submit(credential, payload)` returns a ticket containing a random queue-instance
namespace and a monotonically increasing sequence. A ticket from another
instance cannot cancel current work. `cancel(credential, ticket)` re-verifies
the caller and cancels a pending ticket only when its verified principal matches
the submitting principal. On claim or cancellation, a ticket is no longer
pending; cancellation of running work is the target's responsibility. The
owner must route terminal results or failures back to the client. `close()`
closes admission and returns pending tickets for the owner to mark canceled; it
does not cancel or release work already claimed by the target.

The total byte limit counts payload bytes still waiting in this adapter, and
the item limit bounds each individual payload. The count limit and protected
team positions are separate. In particular, one team's payloads may exhaust the
global byte limit, so another team's first waiter can still be rejected on bytes
even if its count position is protected. The limit excludes caller-held bytes,
Python object overhead, claimed requests, target-owned snapshots, error
tracebacks, device allocations, and traces. A separate owner-controlled budget
must cover all of those and the transition between waiting and target ownership.
The adapter computes waiting-byte totals from bounded per-ticket metadata before
admission and when observed; this scan is linear in the number of waiters.

## Required target contract

The trusted target must advertise `handoff_protocol == HANDOFF_PROTOCOL` and
provide synchronous `accept_next(claim)`. This marker is an integration guard,
not proof that an implementation obeys the contract. The target must:

1. Hold the same broker condition lock used by every competing admission and
   slot-accounting operation. If no global slot or byte reservation is
   available, return `None` without invoking `claim`.
2. Account for a free global preparation slot and conservatively reserve enough
   target-owned host memory for the maximum item before invoking the claim. Under
   the same lock, compute the canonical teams with available per-team slots.
3. Invoke `claim(eligible_teams)` at most once, synchronously on the handoff
   calling thread. Do not retain or call it after `accept_next` returns. A
   `None` claim means no eligible waiter: release any provisional reservation
   and return `None` without accepting a request.
4. For a claimed request, atomically record global and per-team reservations
   before releasing the broker lock, retain or defensively snapshot the payload
   within the target's memory budget, register a terminal outcome owner, and
   return a non-`None` receipt. Adjust a conservative byte reservation to the
   actual payload size without ever leaving the payload unaccounted.
5. If anything fails after claim, record a terminal failure or cancellation for
   that ticket and release the target's reservations exactly once. Retained
   exceptions and tracebacks may themselves keep the request alive; the target
   must bound their memory and retention, too. Apply the same cleanup to OOM,
   cancellation, worker poison, and shutdown.

The adapter never acquires the broker condition while holding its own lock. The
handoff target acquires the broker condition first, and `claim` then acquires
the adapter and queue locks. A verifier is called outside those locks. Lock order
and all other target methods must remain consistent with this arrangement.

`transfer_next()` may return `None` when the target has no capacity or no team
is eligible. On success it returns a ticket and opaque target receipt. A target
exception after claim becomes `HandoffFailure` with the claimed ticket; a fatal
`BaseException` becomes `FatalHandoffFailure`, which is also a `BaseException`.
The dispatch caller must route that ticket to terminal failure, and the target
must clean up its reservations. A callback invoked twice, later, or from another
thread is rejected. These checks cannot protect against a target that lies about
its lock, reservation, exclusivity, or terminal bookkeeping; the target and
verifier are trusted dependencies and require integration tests and review.
