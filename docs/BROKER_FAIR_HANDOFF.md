# Experimental in-process broker handoff

`Broker.enable_fair_handoff` connects one trusted `ExperimentalFairAdapter` to
one broker. It is an in-process service-owner API; it is not integrated with
the IPC server and it does not authenticate callers. The owner must provide a
credential verifier which binds principals to canonical teams, protect that
mapping, and expose only authorized operations to clients. Shared or forged
team labels do not provide identity. The adapter's `transfer_next` and `close`
are service-internal operations.

The broker must be idle when fair mode is enabled. It then permanently rejects
ordinary `submit` calls, so those calls cannot bypass the adapter. A target is
bound to exactly one adapter with the same configured team set and a no-larger
encoded item limit. The caller runs `transfer_next` to move work; this is not a
background scheduling loop. The trusted service owner must also limit concurrent
`transfer_next` calls to at most `max_pending + max_unacknowledged + 1`; this
bounds the preallocated private failure roots if multiple calls fail together.
A transfer takes the broker condition, checks
global and per-team preparation slots, reserves a fixed host commitment, and
claims an eligible waiting team under that same condition. Decoding and
snapshotting happen outside the condition. A successful transfer returns the
waiting ticket and an opaque sequence, not the request handle. The target retains a bounded
recoverable ticket-to-handle entry until the trusted service owner acknowledges
that it has taken separately budgeted custody. `adapter.recover_receipt(ticket)`
looks up the request handle after either success or interruption;
`adapter.acknowledge_receipt(ticket, handle)` confirms that transfer. These are trusted service-internal methods,
not authenticated client APIs. The adapter checks the ticket namespace; the
target is permanently bound to one adapter.

A direct-submit handle can report a terminal outcome before its broker-owned
input root has been retired. Enabling fair mode during that interval raises
`BrokerUnavailable`; the service owner can retry after retirement. There is
not yet a public blocking wait-for-idle API for this transition.

`handle.wait_for_retirement(timeout)` is an exact-handle signal for fair
handoff: it returns only after the broker's execution and job aliases are
retired, raises `TimeoutError` while retirement is pending, and reports
`BrokerUnavailable` when quarantine makes cleanup uncertain. The receipt can
still retain the handle until ACK. A guardian can wait for retirement, ACK,
drop its own handle, outcome and transport aliases, and then release its
separately budgeted result custody. Calling ACK earlier is allowed but does
not establish retirement or dispose the caller's copies. Direct-submit handles
do not provide this retirement signal.

`HandoffFailure` and `FatalHandoffFailure` identify the ticket when the adapter
knows which waiter was claimed, but do not prove the job failed: publication
may have succeeded before the handoff was interrupted. The owner must first
recover and observe or cancel any published handle. If no handle exists and
the broker is healthy, preparation failed before publication; if the broker is
failed, the outcome may be unknown and needs quarantine and reconciliation.
An unexpected exception inside the queue claim can leave even the ticket
indeterminate. The adapter then closes itself to new work; the owner must call
`close()` and reconcile all returned outstanding tickets, deduplicating any
known terminal outcomes. A fatal interruption remains fatal.
For an uncertain active handoff, the public exception is sanitized so that
clearing its traceback cannot clear a privately retained worker exception.
`transfer_next()` wraps a sanitized `KeyboardInterrupt` or `SystemExit` in
`FatalHandoffFailure`; the built-in category is visible in its cause. The
private traceback is not exposed through that cause.

`cancel(credential, ticket)` works only while the ticket is waiting. After a
claim and before a handle can be recovered, decoding is non-preemptible and
there is currently no owner-authorized cancellation interface for that
interval. Once the handle is recovered, its existing pending/running cancellation
semantics apply. Service shutdown must close the adapter separately and close
the broker. After an orderly adapter close, its returned waiting tickets can be
reported canceled; after an uncertain claim, they require reconciliation first.
Shutdown waits for in-flight preparation; a stuck decoder can therefore delay
it. No end-to-end cancellation or network fairness claim is made for this
prototype.

## Memory commitments and assumptions

Each target owns separate input and result `HostBudget` ledgers. A fixed input commitment per claimed
job remains charged while it is preparing, queued and running, and is released
only after the broker drops its input references. The configured commitment
must be at least the sum of the maximum encoded item bytes, the owner-declared
maximum decoder expansion, and three times the worker's per-snapshot input
tensor limit, plus twice its output limit for the model logits and CPU copy.
This intentionally covers overlapping copies in the current resident/budgeted
worker stack; it may conservatively overestimate a specific request. These
limits are per tensor or snapshot, not an aggregate model-workspace cap. A
poisoned or uncertain cleanup retains the relevant commitment and requires
replacing the owner rather than reusing its capacity.
Cleanup that commits its detachment under the broker condition before a later
quarantine may finish disposal after that quarantine. If quarantine wins the
locked decision, staged owners and both commitments remain retained.

Before claim, a second ledger reserves a configured per-result byte commitment
at least as large as the worker's enforced output limit and a bounded receipt
position. The broker checks returned logits storage against
that limit. A completed handle can pin its result, so the commitment remains until
both the owner acknowledges custody and the broker has dropped its own result
aliases. ACK before completion does not release it early. Unacknowledged receipts
can block further admissions even after jobs complete; the owner must drain them.
If retirement or acknowledgement is interrupted or quarantined, preserve the
recipient's separate commitment until it can be reconciled or the process is
replaced.

These are **declared cooperative commitments**, not an operating-system memory
limit. The decoder must be trusted, enforce its declared expansion bound, and
not retain input aliases or traceback frames after return. It must reject
malformed payloads by raising; if it returns an object that fails the supported
forward-request boundary, the broker retains that unknown object and requires
process replacement. A decoder must establish safe disposal of its own partial
inputs before an ordinary exception. Decoder out-of-memory errors and fatal
interruptions are quarantined with their private traceback and commitments
retained until process replacement. The worker must be
trusted to use at most the declared three overlapping input snapshots, stop
using them before a non-quarantined return, and avoid returning output aliases
of input storage. A different implementation needs a new bound and review.
The model must also be trusted or profiled to respect the output and workspace
assumptions: the worker checks logits after the forward returns, so an oversized
model output or intermediate can already have exceeded these declarations.
Python object overhead, allocator fragmentation, caller-held bytes, waiting
adapter storage, arbitrary model intermediates, exception retention, IPC serialization,
other brokers, and GPU memory are outside these ledgers and need their own
limits. Results held by the recipient after ACK need a separate recipient budget.
In particular, one broker's budgets are not machine-wide memory caps.
The worker's `poisoned`, `max_input_bytes`, and `max_output_bytes` getters
are trusted, quick, and nonblocking: they are read under the broker condition.
A getter that blocks can stall service operations; a getter that raises is
quarantined and its error is sanitized.

Waiter positions and queued-job turns are round-robin by configured team, with
the eligibility and liveness conditions described in
[FAIR_ADMISSION.md](FAIR_ADMISSION.md). This does not guarantee fairness of GPU
time, actual work, byte admission, or completion latency. Under full budgets,
a stuck decoder or job can prevent progress. Per-team credential binding and the
fairness handoff must be integrated into the service before treating these
properties as an end-to-end multi-process guarantee.
