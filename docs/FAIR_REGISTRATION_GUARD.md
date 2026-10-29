# Recoverable fair-waiter registration

This is a trusted, in-process adapter API. A service owner first reserves and
roots its own request/custody record, then creates and retains a guard before
queue mutation:

```python
guard = adapter.registration_guard(credential, immutable_payload)
# Store the exact guard in the already rooted request record here.
guard.submit()
ticket = guard.adopt()
```

Creating the guard authenticates the principal with the configured verifier;
it does not queue a request. The verifier and mappings are trusted service
configuration. Guards and tickets are not wire credentials, and untrusted
clients must not be given direct access to the adapter or its private state.
An optional `team_id=` requires the verified principal to map to that exact
configured team. The issued guard is bound to that adapter instance and principal. A guard
remembers the same ticket even after a later cancel or claim; adoption is an
identity recovery operation, not evidence that the ticket is still waiting.
Submitting the same guard again does not enqueue another job. Recovery after an
interrupted outer return uses the exact guard retained before `submit`.

The adapter roots guards for its bounded set of waiting tickets, and also
roots an in-flight guard while it calls the underlying queue. Creating guards
alone does not reserve waiter positions; the service must independently bound
and fund all guard, payload, decoder and exception lifetimes. On claim or
successful cancellation the adapter drops its waiting root; the service must
retain the guard, ticket, claimed handle and their respective custody for as
long as needed. `registration_roots()` and `inflight_registration` are
trusted-owner diagnostic hooks, not disposal or authentication proofs.

An unexpected interruption during queue insertion, removal or cross-layer
bookkeeping can leave the exact queue state unknown. In that case the adapter
latches `indeterminate`, stops admission and refuses cancellation, transfer and
`close()`. Retain all request, payload, handle and claim roots, stop the
service, and use a process supervisor to replace the affected owner. Do not
label any such waiter canceled solely because a snapshot or drain is unavailable.
Even successful return from cancellation may be interrupted before its caller
observes it; a bridge must record cancellation intent before the call and
reconcile it against exact ticket/receipt ownership. A successful close is
limited to known waiting work and does not retire already claimed work. If its
return is lost, `closed_tickets` recovers the last orderly close's exact ticket
snapshot; this snapshot is not available after an indeterminate transition.

The latch is a best-effort observation; arbitrary asynchronous interruption
can strand a Python lock or prevent the latch assignment itself. It is not a
substitute for an external watchdog or a process isolation boundary. A false
latch is not proof that all resources are safe. Input, output and IPC serializer
allocations need independently bounded custody; payload byte limits here are
logical accounting and do not cap physical memory.
