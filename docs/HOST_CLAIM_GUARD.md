# Recoverable host-claim reservation

`HostClaimGuard` is a cooperative recovery aid for a single host-memory claim.
It does not allocate or inspect tensor storage, enforce an operating-system
memory limit, authenticate an owner, or prove that any buffer is disposed.
Recovery assumes the ledger remains responsive. The
[HostBudget interruption limits](HOST_BUDGET.md) apply: an asynchronous
exception at a CPython lock boundary can strand the ledger lock, so a supervisor
must detect an unresponsive worker and use process-level recovery. Ordinary
cancellation should be cooperative; do not inject asynchronous exceptions
into lock-holding code. The service still has to reserve an adequate measured/conservative envelope
before allocating, pin it for in-flight use, retain it while aliases exist,
and release it only after verified disposal or a completed custody transfer.

Keep the request's existing `HostRequestGuard` before opening its generation.
For a claim, create and retain `guard = request.claim_guard(owner, stage,
nbytes, items=...)` before calling `guard.reserve()`. Guard construction itself
does not register a claim. Once reserved, `guard.adopt()` returns the exact
live `HostClaim`, including if the return from `reserve()` or `adopt()` was
interrupted. A repeat `reserve()` confirms that same live claim; it cannot
allocate a second one. Cancellation blocks new reservations, but a guard may
still recover an existing claim so that its owner can dispose it safely.

The ledger retains a guard only while its matching claim is live; each claim
requires at least one counted item, so the number of ledger-retained guards is
bounded by `max_items`. `active_claim_guards()` supports trusted recovery if
the coordinator's local guard reference was lost. The coordinator must use
unique service-generated owner/stage labels and verify exact request generation
and handle identity under exclusive ownership; labels are metadata, not
credentials. Recovery must first select and retain the exact authoritative
`HostBudget` instance and process epoch: another ledger may reuse the same
request ID, generation, labels, and claim number, none of which authenticate
the ledger. If interruption happened before record insertion after the guard
prepared its handle, the guard cannot be reused or attached to a different
claim. The caller must use a new guard and reconcile accounting first.

After `HostClaim.transfer`, the old guard is retired and cannot adopt the new
owner's capability; the transferred live claim remains charged. Release also
retires the guard, and a guard from a closed request cannot attach to a later
request with the same ID. The guard never releases on cancellation, exception,
or loss of the caller's reference. It addresses reservation handoff only;
transfer and in-flight operation handoffs have their own recovery requirements.
If bookkeeping, ownership, or actual storage lifetime cannot be established,
the service must retain the claim and fail closed rather than infer disposal.
