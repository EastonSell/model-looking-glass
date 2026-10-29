# Bounded fair admission waiters

`looking_glass.fair_admission.FairAdmissionQueue` is a standalone in-process
building block for a broker. It holds a bounded number of requests waiting for
admission and selects one at a time by canonical team. It does not allocate or
reserve broker slots, execute work, enforce byte limits, or authenticate users.
The existing broker does not yet use this component; the integration below is a
proposal, not a claim of current shared-service behavior.

```python
from looking_glass.fair_admission import FairAdmissionQueue, AdmissionEmpty

waiting = FairAdmissionQueue(
    {'researcher': 'analysis', 'researcher-alias': 'analysis', 'reviewer': 'review'},
    max_waiters=4,
    max_waiters_per_team=2,
)
ticket = waiting.submit('researcher', {'work': 'example'})
request = waiting.claim_next({'analysis', 'review'})
assert request.ticket_id == ticket
assert request.team_id == 'analysis'
```

The constructor copies a trusted, nonempty mapping from principal IDs to
canonical team IDs; both must be nonempty strings. The mapping is fixed for the
queue's lifetime. Multiple principals assigned to the same canonical team share
one FIFO queue and quota. Unknown principals are rejected with
`UnknownPrincipal`, a `PermissionError` subclass. Both capacities must be
positive integers, and `max_waiters` must be at least the number of distinct
configured teams. Boolean capacities are rejected.

The broker must authenticate each principal outside this component, and derive
the ID it passes from that verified identity. It must control the mapping and map
all aliases for the same team to the same canonical team. **Caller-supplied team
or principal strings do not prove identity**: this component cannot prevent
impersonation, a malicious trusted mapping, or multiple authorized teams assigned
to the same operator. Do not expose the queue object to untrusted clients. The
broker must authorize cancellation; `cancel(ticket_id)` is a trusted internal
operation and does not itself check a principal's ownership of a ticket.

## Protected slots and dispatch turns

Every configured team with no pending waiter has one protected waiter position.
Positions beyond those reservations are shared, subject to the per-team limit.
For example, with two configured teams, a limit of three, and no review waiter,
analysis can hold at most two waiters. The first review waiter can still enter.
If a team becomes empty again by cancellation or claim, its protected position is
restored. This policy may deliberately leave positions unused when teams are
inactive. When a team already has a waiter, a further submission can receive
`AdmissionFull` (`queue.Full` subclass); finite overload remains rejectable.

Internally, if `n` waiters are present, `e` configured teams have no waiter, and
the global capacity is `C`, the policy preserves `n + e <= C`. A first waiter
for an empty configured team increments `n` and decrements `e`, so it has a
position under the count limits. This guarantee assumes valid authenticated
configuration, `max_waiters >= number of teams`, no additional external rejection
policy, and a caller whose team currently has no waiter. It does not guarantee
that arbitrary further requests are accepted under finite overload.

Each team is FIFO. Active teams take turns in arrival order; a claimed team with
more waiters moves to the back, and newly active teams join the back. The broker
may pass `claim_next(eligible_teams)` to skip teams without compatible capacity;
skipped teams keep their position. An empty eligible set or no eligible waiter
raises `AdmissionEmpty` (`queue.Empty` subclass) without removing work. Unknown
eligible teams are rejected. Once a team's head waiter remains continuously
eligible, newly active teams cannot overtake it; with continued eligible claims,
at most the other configured teams already ahead can each take one turn first.
Earlier waiters in that same team must receive their own turns first.

This is a count-of-claims property, not equal GPU time, bytes, throughput, or a
wall-clock bound. A stuck job, abandoned broker reservation, exhausted device
memory, or a team that remains ineligible can stop progress indefinitely. There
are no internal worker threads, automatic timeouts, priority aging, retries, or
preemption. The queue holds payloads by reference and limits the number of
waiters, not their bytes, caller memory, running jobs, traces, or GPU allocations.
Those need independent broker limits. Methods are thread-safe, with ordering
determined by lock acquisition; concurrent call ordering is not reproducible.

## Cancellation and broker handoff

`submit` returns an integer ticket that is never reused within a queue instance.
`cancel(ticket)` removes a pending waiter and returns true, or returns false
after a claim or earlier cancellation; invalid ticket types, including booleans,
are rejected. A racing cancel and claim have exactly one winner for that ticket.
If tickets escape the process or survive a restart, namespace them with a broker
instance identity and reject stale handles; ticket numbers restart with a new
queue instance. The broker is responsible for cancellation and cleanup after
claim. Payload finalizers run outside the queue's lock, but an external lock may
still be held by the broker.

`claim_next` transfers ownership of the request to the broker; it is **not an
atomic resource reservation**. To integrate it safely, the broker must protect
its capacity checks, the claim, and its own slot-accounting changes with the
same broker condition lock. It can compute the set of teams with available
per-team slots while holding that lock, claim one eligible request, and record
its global and per-team reservation before releasing the lock. The broker must
retain the request and mark it failed or canceled if handoff fails, rather than
silently losing it. Its existing cleanup paths must release reservations on
cancellation, exceptions, OOM, shutdown, and after work leaves the counted stage.
No reservation is handed to an external client by this primitive; therefore it
does not introduce a client-held lease, but it cannot repair a broker-held
reservation that the broker fails to release.
