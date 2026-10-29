# Experimental grouped broker

This is an opt-in, trusted in-process coordinator. It is not wired into the
serial service or IPC. It combines compatible requests into one forward on
an existing `GroupedWorker`; it does not run simultaneous forwards on the
same model. A single dispatch owner calls `run_once`, while teams may submit,
cancel and consume results from separate threads. The queue uses its
round-robin team policy and a bounded coalescing wait. A head that cannot
form an admitted cohort by the deadline completes with
`no_compatible_cohort`; there is no implicit smaller-batch retry.
Relative deadlines use `perf_counter`; actual wakeup latency still depends on
the operating system and must be measured separately.

The owner supplies an exact `BatchProfile`, a `CertifiedBatchPolicy`, an
independently checked live model/runtime identity, and conservative,
service-owned input-snapshot and output-validation reservations. Digests
and profile labels are provenance fields, not measurements or proof of model
identity. The evidence policy only admits the particular row counts and
sampled cases it validates; agreement on those cases does not establish general
batch separability. The underlying worker requires a trusted whole-batch
device profile, synchronizes the forward before handing its results back, and
holds its device admission through splitting. This component neither
profiles allocations nor creates a GPU lease or supervisor.

The `HostBudget` must be empty and cooperatively exclusive to the broker for
its lifetime. The exact `GroupedWorker` must use that ledger and the supplied
profile. Before accepting any jobs, the constructor checks capacity for
`max_pending` input snapshot claims plus
`floor(max_pending / minimum_admitted_cohort_size)` simultaneously retained
groups. Each group is charged for the worker's conservative group envelope
and its largest configured post-split validation reservation; request and
operation slots are also checked. Completed groups continue to occupy their
job slots until every member is disposed. This prevents accepted snapshots
from consuming the ledger capacity needed to form their groups, assuming no
other code bypasses the exclusive ownership contract.

The group envelope covers the specific logical input copies, full output,
split outputs and worker finite-check scratch described by `GroupedWorker`;
the separate validation reservation covers the broker's post-split finite
check. Snapshot storage must be independently owned, contiguous, zero-offset,
and exactly its logical byte size. The caller's input storage, Python object
overhead, arbitrary model allocations, allocator fragmentation, RSS, client
copies and JSON/socket serialization are outside these claims. Every additional
simultaneous copy needs its own conservative reservation before allocation.
No separate ledger can enforce an aggregate process memory ceiling.

`submit(team_id, request)` returns a handle, also recoverable by
`lookup(team_id, request_id)` if the submit return is interrupted. Team and
request identifiers are never reused in the same broker epoch; a configured
lifetime limit bounds retained recovery metadata. The owner must not mutate
inputs concurrently with snapshotting. Request validation and sampled
evidence do not authenticate an untrusted client or checkpoint.

Each result consumer obtains its own bounded ticket with `handle.delivery()`.
The first `ticket.wait(timeout)` binds that ticket to the exact current thread;
only that thread can retry the wait, confirm the receipt or release the
ticket. A timeout retains the ticket and may be retried or explicitly
released by that thread. After a successful wait, call
`ticket.confirm(result)` with the exact returned object, discard every result
and tensor alias, then call `ticket.release()` and `handle.dispose()`. Repeated
or concurrent consumers need separate tickets, and all tickets must be
released before a handle can be disposed. `handle.active_deliveries()`
exposes exact tickets for trusted recovery; an abandoned consumer thread is
not automatically assumed to have disposed its aliases.
If disposal is interrupted, retry it on the same handle: an already-disposed
handle also retries retirement of its exact group. Until that handoff completes
or the process is replaced, the group's capacity remains charged.

A delivered but unconfirmed ticket cannot be released. If a wait was
interrupted before its result reached the caller, the same consumer may call
`ticket.abandon()` only after establishing that the wait has stopped and all
exceptions, traceback/context graphs and other possible result aliases have
been discarded. This is a cooperative disposal attestation, not an ability to
inspect or revoke arbitrary Python references. If that fact cannot be
established, retain the ticket and use supervised process replacement.
Never abandon a result merely because the delivery call raised.

The cohort state serializes cancellation with its terminal publication. A
cancellation accepted before publication suppresses that member's public
output; a later cancellation returns false. An accepted prelaunch
cancellation prevents the whole cohort from launching. A failure after launch,
including out-of-memory, takes precedence over individual cancellations for
every member and quarantines the owner. The shared forward is not cancelled
by one member. Unexpected or ambiguous handoff failures also quarantine and
retain the affected claims and exception state; public coordinator errors do
not expose the private exception context. Quarantine requires the external
supervisor to retain the worker, ledger and device ownership until a safely
reaped process; it must not declare cleanup merely because a method returned.
The supervisor should monitor both the nonblocking `failed` and `quarantined`
properties: a failed worker can be unavailable even if its device was drained.
`unavailable` also signals a pending health decision; it is not itself a
permanent quarantine. A supervisor must not certify a clean owner while this
state is unresolved. A dispatch timeout bounds waiting before a cohort is
claimed. Once a snapshot or cohort has been staged, a pending health probe
can delay progress until it resolves; the associated claims remain charged.
An unresponsive probe requires the external watchdog to replace the process.

The optional session-bound `GroupedWorkspaceWorker` has a separate binding
transition from the serial service. Broker construction holds its preflight
gate and checks that the exact worker has no active control or untransferred
producer result; a busy owner returns `HostBusy` before binding. This does not
certify a GPU profile or enable timing observations. After admission has been
closed and the broker, worker and resident are closed, the trusted owner may
call `retire_grouped_owner(worker, broker, callers_quiescent=True)` only after
joining all dispatch, control, delivery, cancellation, disposal and health
callers and disposing every associated alias and exception graph. The helper
checks the exact binding, empty ledgers and custody slots before permanently
detaching the broker and worker. It retains no proof of external or native
alias disposal, and it does not release the physical lease or either fixed
device commitment. The capabilities and relevant roots must remain held
through process exit. A separately reviewed one-shot may release inert
accounting commitments only after verified drain, zero allocator counters and
host claims, and disposal of model/input/output, trace and callback custody,
under the still-held physical lease, with no further CUDA work before clean
attestation and exit. This helper does not perform that additional cleanup.
Uncertain retirement requires process replacement.

`close()` releases the exclusive ledger only after all healthy jobs,
deliveries and groups are disposed and no dispatcher remains active. It does
not close the model owner or release the GPU lease. Arbitrary repeated
asynchronous interruption can strand Python locks or interrupt the latch
machinery itself; a watchdog must replace an unresponsive process rather than
assuming an in-process cleanup completed. Nothing here guarantees safety
against arbitrary mutation of these trusted in-process objects.
