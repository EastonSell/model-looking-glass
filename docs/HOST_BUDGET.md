# Host-memory custody ledger

`HostBudget` is an in-process, thread-safe ledger for **declared** host-storage
commitments. It is a building block, not yet a host-memory limit for the
worker, trace store, broker, or transport. Those components must reserve before
allocating and participate in the handoff and release protocol before any
aggregate limit can be claimed. Its lifetime is independent of the GPU lease:
a CPU result can remain live after GPU execution has ended.

Configure a byte cap, an item cap, and separate caps for open requests and
in-flight operations. Every claim consumes at least one item, including a
zero-byte claim. A claim represents one independently disposed buffer or a
conservatively held group of buffers; claims cannot be split. When source and
copy coexist, reserve both. Do not subtract the source merely because a clone
has been requested. Reserve each additional list, string, encoded payload, or
send buffer when that allocation is part of the policy. A trusted owner must
choose conservative estimates; the ledger cannot infer Python object overhead,
allocator slack, copies made by other code, kernel socket buffers, other
processes, or actual operating-system RSS. A separate ledger per process does
not impose a machine-wide aggregate limit. Snapshots and `active_*()` results
allocate caller-owned tuples that are not themselves charged to the ledger.

```python
from looking_glass.host_budget import HostBudget

budget = HostBudget(max_bytes=32, max_items=2, max_requests=1, max_operations=1)
request = budget.open_request("run-1")
source = request.reserve("worker", "logits", 16)       # before allocation
clone = request.reserve("store", "trace-copy", 16)     # before copying
operation = request.begin_operation("clone", [source, clone])

# Perform the copy. Report completion only after the underlying work ends.
still_live = operation.finish()

# Dispose the actual source and clone first; these calls only release accounting.
source.release()
clone.release()
request.close()
```

For a coordinator that must recover an interrupted request handoff, create and
retain a guard **before** registering anything:

```python
guard = budget.request_guard("run-2")  # no ledger request or claim yet
guard.open()                           # stores the exact handle before registration
request = guard.adopt()                # returns an alias; guard still retains it

# Reserve and use claims as above; dispose and release them explicitly.
guard.cancel_and_close_empty()         # cancels, then closes only if not busy
```

Each registered guard is retained by its request record and can be recovered
with `active_guards()`, so its ledger-owned metadata is bounded by the request
limit and retires when the request closes, including a direct `request.close()`.
If registration or adoption is interrupted, the retained guard identifies its
exact handle and generation; it cannot operate on a later request reusing the
same ID. If interruption happened before record insertion after the guard
prepared a handle, abandon that guard with `cancel_and_close_empty()` and
create a new one. `open()` is idempotent while its original generation is live;
a guard cannot issue its old handle as a new generation.

Recovery depends on the ledger lock remaining accessible. Asynchronous
exception injection at arbitrary CPython bytecode boundaries can interrupt a
`with RLock` exit before the lock is released, leaving other threads blocked;
this is a general limitation of the ledger's lock regions, including this
guard. Do not inject asynchronous exceptions into lock-holding code. A
signal-delivered `KeyboardInterrupt` can also arrive asynchronously; ordinary
cancellation should use the cooperative cancellation API. A
supervisor must detect an unresponsive worker and use process-level recovery
instead of assuming `active_guards()` or other in-process inspection will
always return. Any external buffers and claims require their own verified
cleanup; process termination alone does not prove their disposal.

`cancel_and_close_empty()` tombstones the matching request, then closes only
when no claims or operations remain. A busy close raises `HostBusy` and leaves
every commitment charged; the guard never finishes operations or releases
claims. The coordinator must establish completion and dispose real storage
before explicitly releasing claim handles. This guard does not provide durable
cross-process identity or guarantee recovery if callers bypass the ledger or
lose the underlying process.

An operation pins **all** listed claims until `finish()`; a shared source can
participate in multiple operations, and it cannot be transferred or released
until all of them finish. The operation owner must call `finish()` only when
the copy, serialization, or send has actually stopped and any required
completion or disconnect has been established. A claim's `release()` is an
attestation that its covered storage and aliases were disposed, not an action
that frees Python tensors or external copies. The trusted coordinator can
recover outstanding handle objects using `active_requests()`,
`active_claims()`, and `active_operations()`. Losing a handle or allowing it to
be garbage collected does not automatically free its commitment.
`finish()` is single-use and rejects a second call; `release()` and `close()`
are idempotent for their current, issued handles.

`claim.transfer(new_owner)` moves custody without changing committed bytes or
items and invalidates the previous handle. The ledger records the bounded
owner history, stage, claim ID, request ID and request generation, but no
payload. Treat IDs and owner labels as metadata; do not put prompts or other
secrets in them. The generation distinguishes reuse of an ID within one ledger
instance; it is not a durable cross-process or cross-restart identifier. A
configured transfer-history cap bounds metadata. One additional cleanup-only
transfer is reserved after cancellation, even if ordinary transfer history is
full.

Cancellation is a tombstone: it blocks new reservations and operations, but
does not release a claim, stop a copy, or prove that any buffer was disposed.
Existing operations remain pinned until their owners confirm completion. A
normal custody transfer is rejected after cancellation; one explicit
`transfer(new_owner, cleanup=True)` can hand the claim to a cleanup owner.
The request can close only after all claims and operations have retired.
`operation.finish()` returns whether the request was live **at that instant**;
cancellation can occur immediately afterward. A store or transport must use
its own atomic publish-versus-cancel protocol. A `True` result alone is not
authorization to publish or send a result.

The handle checks and lock coordinate cooperating code in the same Python
process. They do not sandbox callers with access to ledger internals, force
storage disposal, or authenticate a remote client's acknowledgement. On
exceptions, do not release a reservation merely because a function unwound:
verify the relevant work and storage have ended. Registry creation and handoff
attempt rollback for exceptions caught inside the transaction. After ambiguous
completion, including interruption around return, lock release, or rollback,
the trusted coordinator must inspect and recover outstanding handles. Failed
cleanup should leave the claim charged until a
responsible owner completes disposal or the execution process is safely torn
down.
