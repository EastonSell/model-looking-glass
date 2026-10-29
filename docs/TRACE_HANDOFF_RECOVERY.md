# Supervised recovery of a quarantined CPU handoff

This experimental, process-local helper can discard a narrowly eligible
quarantined `SealedBatchRegistry` handoff after a trusted supervisor has
established local quiescence and disposal. It is not connected to `Broker`,
IPC, a GPU, a remote acknowledgement, or an authenticated transport. It
does not resend payloads, create a complete marker, establish source
provenance, or prove a physical memory bound.

Recovery is eligible only for an exact, still-rooted bridge which accepted a
recipient, whose job is still a sealed `CpuTraceOwner` job, whose manifest
request and claim are known, and whose bridge operation has stopped. A
quarantined construction, failed or interrupted cleanup, missing claim,
tombstoned manifest request, active call, or changed owner requires separate
supervision; this helper does not infer how to repair it. A process-lifetime
supervisor must retain strong references to the registry, bridge, owner, job,
budget, and affected capabilities through reconciliation or verified process
termination. Object collection is not disposal evidence.

The same trusted non-retaining sink contract described in
[TRACE_BATCH_HANDOFF.md](TRACE_BATCH_HANDOFF.md) applies on success **and**
failure. A child may release its own serialization claim during failure
cleanup before recovery starts. If a callback, exception attribute, nested
exception group, traceback, cycle, or external alias can retain bytes, that
owner must reserve and retain a separate claim before the copy exists.
Recovery cannot retroactively account for a previously unclaimed copy.

## Procedure

1. The trusted supervisor records a new 32-hex-character `recovery_id` and
   calls `register_recovery()` with the exact broker ID, process generation,
   job ID, registration ID, job generation, transfer ID, and rooted bridge.
   The exact registry instance and bridge capability remain part of that
   binding. The caller must serialize recovery registration and never reuse a
   recovery identity in a way that could confuse its own supervisor. Only one
   recovery can be rooted on a bridge. If the registration return is lost,
   `lookup_recovery(bridge, recovery_id=...)` recovers that same object.
2. The caller has already stopped new uses of the bridge, job, child, sink,
   and recipient aliases. `bind()` reserves an additional logical inventory
   claim, pins it together with the exact manifest claim under the same host
   request generation, then records the original slot-to-child capabilities.
   The child set is private; it does not give the supervisor a new payload
   delivery capability. `bind()` returns `False` on a known capacity denial
   before inventory copying, after rolling back any new inventory claim. A
   lost or uncertain reservation, pin, or rollback outcome remains rooted and
   quarantined. A cancelled manifest request cannot start these ordinary
   ledger operations and needs process-level reconciliation.
3. `scope()` returns a pinned metadata description containing the exact
   job/process/transfer identity, manifest request generation and claim ID,
   inventory claim and operation IDs, and each original event's slot,
   delivery and sequence identity. The supervisor verifies this against its
   independently retained authority, then calls `ack_scope(scope)`. The
   acknowledgement verifies exact types and values; it authenticates nobody
   and does not attest payload receipt, completion, or disposal.
   An exact duplicate scope acknowledgement is idempotent if its return is
   interrupted.
   No static list of child claims is treated as authoritative: each exact
   child retains and drains its own live request generation, claims, and pins.
4. The supervisor drops the scope and all recipient, sink, metadata, payload,
   exception and traceback aliases, or proves that every surviving copy has
   separate custody. It must also establish that **all** old bridge and child
   calls have unwound and that no other owner can start a new one. Only then
   call `confirm_disposed()`. The attestation is a trusted statement; an
   inactive bridge flag, missing reference, garbage collection, receipt or
   release ACK, disconnect, timeout, or child drain cannot prove it. Scope
   cannot be exposed again after this attestation. If its return is
   interrupted, inspect the exact rooted recovery's `state`; an `attested`
   state means the attestation committed and must not be replayed blindly.
5. `drain(timeout=...)` verifies the frozen exact slot/child capabilities and
   invokes the existing job and child cancellation/drain outside the bridge
   control lock. A `False` result means only that disposal is incomplete:
   the child may be busy, timed out, or quarantined. The metadata claims and
   operation remain held; retry can be attempted without redelivery. A slot
   missing after an earlier partial drain is accepted only when the original
   slot is marked retired and the original child independently reports a
   completed drain. Unexpected replacements or unproven absence quarantine
   recovery. The timeout is passed to the underlying child drain; it is not a
   wall-clock bound on recovery. Lock acquisition, synchronous cancellation,
   finalizers, owner retirement, and ledger operations may block separately.
6. After all children drain, the exact job is retired and the bridge-owned
   metadata and private inventory are dropped outside the control lock while
   the operation remains pinned. Recovery finishes the pin, releases its
   inventory and manifest claims, closes the exact request, and marks the
   bridge disposed. The supervisor can then retire the bridge. If the final
   successful return is lost, the rooted recovery's `state` and bridge's
   `state` can be inspected before retirement.

An exception or ambiguous transition after bind or drain has been admitted
leaves the recovery `uncertain`, roots the error, and retains any remaining
claims for supervision, except when an already committed return has the
explicit recovery behavior described above. A rejected precondition or
incorrect scope acknowledgement does not itself change the live recovery.
A pin or finish return that is lost is not presumed absent or complete.
Arbitrary asynchronous exceptions can also strand Python locks; an
unresponsive process must be handled by the surrounding supervisor. No
automatic recovery of uncertain ledgers or arbitrary child quarantine is
provided.

## Accounting scope

The additional inventory reservation is `(3 * event_count + 4) * 65536`
logical bytes and `3 * event_count + 4` items, with at most 256 events. This
is a cooperative accounting envelope, not a measurement or limit on Python
object sizes, allocator overhead, process RSS, GPU memory, or remote copies.
The original manifest claim remains independently charged; both claims are
pinned while inventory and cleanup operations can use them. Individual
deliveries separately account for their own sources, store copies and
serialization operations. A proven pre-copy capacity denial is retryable
with the rollback described above, but `bind()` may block acquiring locks.
No atomic whole-job admission or fairness guarantee is implied.

Recipient processes, sockets, separately held outputs and externally backed
buffers are independent custody domains. Neither a local drain nor termination
of this process proves that surviving copies in those domains were freed.
