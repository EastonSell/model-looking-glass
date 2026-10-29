# Experimental trace-delivery custody boundary

Status: **CPU-tested, trusted in-process prototype. Not a network protocol or a
broker/service feature.** The existing IPC service remains forward-only and must
not admit trace jobs using this adapter. Its tests do not establish GPU safety,
machine-wide RAM limits, remote authentication, throughput, or a bound on the
memory of consumers.

`TraceDelivery` stages one event for one sample. It takes trusted individual
identity fields for the broker lifetime ID, job, team, run, request, sample
and batch row, constructs a `DeliveryIdentity`, and creates a fresh random
delivery generation.
It checks those fields against the supplied `CapturedSlice`, and binds receipt
and release acknowledgements to the complete identity. Only the owning
coordinator can decide whether the identity corresponds to an authenticated
principal. This class does not authenticate a caller. It also does **not**
check the frozen allowed token positions or trace-site plan: the trusted owner
must validate both, along with all batch/model postchecks, before publication.
No full-batch result or other sample is returned by the adapter.
The capture's job ID must be `lg-{broker_id}-{job_id}`; this distinguishes
broker lifetimes if schema metadata is exported without its delivery envelope.

## Required ownership sequence

1. Before capture, construct a delivery with trusted limits and call
   `reserve_source(worst_case_bytes)`. A producer must hold this reservation
   throughout capture and must exclusively own the mutable source while it
   is staged. A separate producer envelope is required for forward inputs,
   activations, GPU work, index/select temporaries and any other source or
   workspace. If the producer uses another ledger generation, conservatively
   reserve both envelopes before capture and keep both until their respective
   owners actually dispose their references. This double reservation is
   intentional and may reject otherwise feasible work.
2. Call `stage(slice)` with a plain, detached, contiguous CPU float16,
   float32 or float64 tensor. Its backing storage must be exactly the payload
   size, and shape, dtype and sample identity must match metadata. Source and
   saved dtype must agree; source-device provenance remains trusted producer
   metadata. Store and aggregate destination claims are obtained before copying. Both the source
   and destination claims remain pinned until the copy has stopped. No
   concurrent mutation is allowed. This is a trusted-producer contract:
   the adapter cannot prove when capture occurred or detect all aliases.
3. After staging and **after the producer has discarded every source tensor,
   slice and alias**, call `confirm_source_disposed()`. The adapter clears
   its own source references before it can release that claim. Confirming
   while a copy is running records the declaration but the source claim remains
   pinned until the copy exits. An untrue declaration breaks the accounting
   contract. If capture is abandoned, cancel and confirm disposal of any
   source that was reserved before draining.
4. The coordinator validates the full frozen request/site plan and completes
   all worker/batch postchecks, including any device synchronization required
   upstream, then calls `publish()`. This exposes immutable metadata only;
   it is not evidence that the payload reached a consumer, that a multi-event
   batch is complete, or that a trace explains model behavior. Publication
   commits at its final cancellation-signal check under the lifecycle lock.
   If that check precedes the cancellation signal, the manifest may be
   returned even if cancellation returns before publication does. An already
   returned manifest cannot be retracted.
5. `send(sink)` first reserves a separate logical serialization claim and
   pins it with the store/delivery claim through `take()`, serialization and
   synchronous sink completion. The same coordinator retains the aggregate
   destination claim across `take()` even though the store's own quota ends.
   The sink receives exactly one little-endian contiguous payload in the
   manifest's saved dtype/shape. It must complete the entire local send
   synchronously and return `None`, or raise on any partial/failing send. It
   must not retain the buffer, copy it for later use, or retain traceback
   frames containing it. These conditions cannot be enforced for arbitrary
   Python callbacks. A future, generator, async generator or awaitable returned
   by a sink is treated as uncertain deferred work and quarantines the claims.
6. An authenticated future transport may call `ack_received(identity)` only
   after it has established that a complete payload arrived. Receipt keeps
   the service payload charged. A separate `ack_release(identity)` schedules
   local disposal; exact duplicate receipt and release acknowledgements are
   idempotent while the generation remains active. A wrong or stale identity
   never releases a claim. The acknowledgement is a cooperative statement, not proof that a
   client discarded its copy or refrained from making more copies.
7. `drain(timeout)` returns true only after local operations, cleanup,
   producer confirmation and ledger closure complete. It includes the wait
   for the lifecycle lock and returns false on timeout or quarantine. It does
   not stop an in-flight operation. `close(timeout)` requests cancellation
   before calling drain; its total elapsed time is not bounded by the drain
   argument if scheduling cancellation itself is delayed.

For the abandoned/cancelled path, `cancel()` first sets a signal without
acquiring the lifecycle or ledger locks, then schedules ledger tombstoning on
a worker. It does not wait for the ledger tombstone, store copies, or the sink.
A copy already in progress may finish, but no publication may begin after
the signal. Send commits at its final cancellation-signal check under the
lifecycle lock; if that check precedes the signal, the sink may start or emit
bytes even after `cancel()` returns. Its work remains charged until it returns and
the partial payload must not be accepted as complete. Store reservation,
take, drop, tensor destruction and final ledger closure are outside the
lifecycle lock; cleanup holds the relevant claims until local disposal ends.
There is at most one cleanup worker per active generation. A blocked cleanup
therefore makes drain return false, not a false completion.

This single-event signal-check rule does not implement the request-level atomic
publication transaction proposed in [TRACE_CUSTODY_PLAN.md](TRACE_CUSTODY_PLAN.md).
Publishing several individual events cannot seal a multi-event batch: a future
coordinator must establish its own atomic manifest/cancellation decision after
all expected events and postchecks are complete.

The owner must treat `state == "quarantined"` or a failed drain as unresolved.
On uncertain handoff, deferred sink, failed ledger transition, or failed
disposal, claims are deliberately retained rather than guessed away. A
supervisor must isolate that execution process and retain the associated
resource accounting until cleanup is verified or the actual process has
terminated. This prototype has no recovery API for a quarantined delivery.
Invocation guards quarantine exposed uncertain handoffs, but Python cannot
promise atomic cleanup under arbitrary interruption at every bytecode.
Construction uses a host-ledger guard bound to its exact request generation.
If construction fails, its payload-free metadata claim is released only after
that exact generation is tombstoned and its acquisition-specific owner token,
stage, size and untransferred ownership are checked. The trusted
coordinator must give the constructor exclusive ownership of that generation
until construction or its cleanup has stopped; other code must not adopt,
reserve, transfer, release, or recover its claims concurrently. If cleanup
itself fails, no delivery is returned: `DeliveryConstructionError.guard`
identifies the exact guard for trusted supervisor reconciliation. If even the
error cannot be constructed or delivered, `HostBudget.active_guards()`
can expose outstanding guards but does not identify the failed one by itself;
stop and reconcile affected work, or use process-level recovery rather than
guessing which generation can be released.

## Scope of the declared bounds

The shared `HostBudget` limits the **declared logical bytes**, item count,
request count and operation count for participating deliveries. The separate
`HostTraceStore` limits its own store payloads and pending copies. Source,
store and serialization claims are distinct when those buffers coexist.
`max_payload_bytes` bounds each single-event payload and its binary encoded
copy. Claims and reservations do not intercept allocation, cap RSS, or count
Python objects, allocator rounding, NumPy view metadata, library workspaces,
socket/kernel buffers, or unknown aliases. The trusted coordinator is
responsible for other participants' claims and any conservative estimates.

Before metadata construction, the adapter reserves a minimum 65,536-byte
logical encoded envelope. The bounded schema has at most eight dimensions,
128 token indices and eight output-path components; each bounded text component
contains at most 128 code points and JSON escapes at most 12 ASCII bytes per
code point. Even with 19-digit indices and fixed keys the encoded event fits
within the envelope; the code checks its actual JSON length. This envelope
bounds the representation, not transient Python object overhead. A returned
manifest belongs to the caller; caller-retained metadata is outside the
adapter's ledger after local closure.

The generation ID is random and scoped to a broker lifetime. A production
trace wire protocol still needs authenticated identity binding, bounded
framing and total queued deliveries, explicit complete-manifest/final markers,
partial-frame rejection, broker-owned result disposal, disconnect handling,
and an explicit client lifetime contract. Neither send completion, ACK, nor
disconnect proves that a peer released memory. The current interface does not
provide those features and must not be exposed as a trace endpoint.

Run the CPU tests with:

```powershell
python -m unittest discover -s tests -p test_trace_transport.py -v
```
