# Sealed CPU trace-batch handoff

`SealedBatchRegistry` is a trusted, process-local prototype for handing one
complete sealed `CpuTraceOwner` job to one recipient. It roots the exact job
and its host claims while a recipient receives the metadata and each selected
payload. A complete marker is available only after the recipient has attested
to the manifest and every payload. This module is not connected to `Broker`,
IPC, a GPU, or a wire protocol, and it does not authenticate a recipient.

The owner must authorize the chosen recipient for every sample and all
metadata in the job. All callers, sinks, and model/trace producers are trusted
cooperators. A cooperative fence prevents two registries in this module from
registering the same live job concurrently; it does not stop an independently
held `CpuTraceOwner`, job, batch, or child alias from bypassing this handoff.
The supervisor must provide exclusive lifecycle ownership until the handoff is
retired. Treating this API as a Python isolation boundary is unsafe.

## Identity and registration

The supervisor supplies the broker ID, process generation, transfer ID,
recipient ID, and the existing job's registration ID and generation. The
registry roots the exact job and permits only strictly increasing generations
across registrations within that registry instance. The caller must preserve
non-reused broker/job IDs and process, registry, registration, and transfer
epochs across replacement or restart; separate registry instances do not
provide a global allocator. Schema event identity `(run_id, sequence)` and
forward-index allocation remain separate trusted cross-job responsibilities.

`lookup(job_id, registration_id=..., generation=..., transfer_id=...)`
selects the rooted handoff by the caller's already-recorded identity. A
registration interrupted before insertion rolls back its generation and may
be retried with the same live job. If its return is interrupted after
insertion, look up the exact key before deciding what to do. General
asynchronous interruption can still strand a Python lock or make an outcome
uncertain; a process-lifetime supervisor must retain the owner, registry, job,
budget and capabilities until reconciliation or verified process termination.

Each `EventKey` also binds the exact child slot generation, full
`DeliveryIdentity`, and event sequence to the job and transfer. A key is a
trusted local routing capability, not authentication or a historical
uniqueness guarantee for the child delivery UUID alone. Stale and mismatched
keys are rejected. The underlying trace plan and sealed manifest still rely
on the producer's documented provenance and trusted model/input binding.

## Cooperative sequence

1. Register an already sealed job and recover it by its exact key. Call
   `prepare()` before exposing a recipient; it reserves and pins a logical
   manifest envelope before constructing the bridge-owned metadata copies.
2. Call `offer()`, then `accept(recipient_id=..., transfer_id=...)`. A lost
   accept return can be recovered with `recover_recipient()` using the same
   identity; it returns the same recipient object rather than duplicating
   custody. The bridge and recipient are non-copyable capabilities.
3. The recipient calls `manifest()`, then `ack_manifest()` to attest that it
   received the complete metadata. It can obtain routing keys with `events()`;
   `send(key, sink)` is permitted only after the manifest acknowledgement.
   The sink must synchronously consume the bytes, return `None`, and retain
   no payload or deferred work. A copying or retaining sink is outside this
   contract and requires a separate owner and claim for every surviving copy.
4. After a successful send, `ack_received(key)` records that payload's receipt.
   `complete()` returns a marker only once the manifest and every expected
   payload have been acknowledged. Each `ack_release(key)` is then a
   cooperative release attestation for that delivery. Exact duplicate ACKs
   are idempotent; a completed marker remains recoverable after individual
   releases until the handoff starts disposal. If a send's return is lost
   after its local `sent` state committed, `delivery_state(key)` recovers that
   outcome without resending. `sent` means only that the trusted synchronous
   sink returned successfully; it is not a receipt or disposal attestation.
5. The recipient must drop every alias it holds to the manifest, event keys,
   marker, callback frames, exceptions and their nested tracebacks, or arrange
   separate custody for any permitted retained object. Only then may it call
   `confirm_disposed()`. The owner calls `dispose(timeout=...)`, retries a
   normal incomplete drain if needed, and finally retires the bridge.

The marker represents cooperative receipt of one complete sealed manifest and
all of its payloads. It is not a hash, source-provenance proof, authenticated
wire frame, or proof that the receiver has freed any memory. A receiver must
stage partial results until it observes the marker. A send can have exposed
bytes before a later send or cancellation fails; the marker does not retract
those bytes. The bridge does not retry a failed or interrupted sink, because
the amount delivered may be unknown.

Receipt, release ACK, timeout and disconnect are not proof that client or
other-process copies have been destroyed. Process termination can establish
only disposal within that process, not disposal of copies held by a surviving
peer, socket, or caller. Those owners require separate claims and their own
reconciliation. This in-process prototype does not implement remote ACKs.

## Accounting and failure handling

At most 256 events are accepted per registered job. Before metadata copying,
the bridge reserves `(2 * event_count + 1) * 65536` logical bytes and
`2 * event_count + 1` items for the two bridge-owned manifest representations
and a workspace envelope. This is a declared cooperative accounting amount,
not a physical Python allocation, allocator, or RSS bound: tuples, keys,
objects, JSON temporaries and any recipient copies can have different memory
costs. A future wire serialization or copied manifest needs its own
appropriately bounded reservation. Model residency, external inputs and other
processes are accounted separately as described in `CPU_TRACE_OWNER.md`.

Each underlying `TraceDelivery` holds its own destination claim and obtains
and pins a separate serialization claim before allocating its send buffer.
This prototype does not guarantee atomic admission of all future delivery
claims. If the exact child reports `HostBudgetExceeded` during its documented
pre-sink admission, no sink was invoked, the slot remains pending, and the
caller may retry after capacity becomes available. Other errors after a send
is admitted, lost returns before the `sent` commit, and sink failures are
ambiguous and quarantine the handoff. Invalid keys or states rejected before
admission leave the live handoff usable. A lost return after the commit is
recoverable using the exact-key state read above. The bridge keeps its
metadata claim rooted, and local in-flight or uncertain copies remain
charged. A child may release its own claim after it establishes that its local
payload is disposed. A capacity failure before any manifest copy or transfer
can be cleaned up with `abort_before_transfer(timeout=...)`.

Cancellation is cooperative. `disconnect()` promptly sets a local signal,
then signals the active child and underlying job outside the control lock.
Before any recipient is accepted, an owner may use `abort_before_transfer()`;
a normal drain timeout is retryable while claims remain held. After transfer,
disconnect or uncertain cleanup quarantines the bridge for supervision; it
does not automatically retire the bridge or release its metadata claim. A
child's asynchronous cancellation may dispose a quiescent local payload and
release that child's claim. `timeout` bounds the later drain waits, not
necessarily synchronous child signalling. Do not call cleanup on a
latency-sensitive control thread.

Control-lock checks serialize the complete-marker decision with the local
cancellation state, but overlapping calls can return in the opposite order:
an operation whose decision occurred before cancellation may return after
`disconnect()` starts. Cancellation cannot revoke bytes or a marker already
handed to a recipient. ACK and registration return recovery described above
does not make all Python interruption points recoverable: in particular, an
asynchronous exception while exiting a lock can leave the lock unusable. An
unresponsive or ambiguous bridge requires process-level supervision; garbage
collection is never evidence that charged resources or aliases were disposed.
