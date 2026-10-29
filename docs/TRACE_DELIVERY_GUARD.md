# Experimental delivery-slot adoption

Status: **trusted process-local CPU prototype**. The existing service and IPC
remain forward-only. This is not an authenticated protocol, batch seal, GPU
admission mechanism, durable registry, or automatic ownership/disposal system.
It does not establish a bound on process RSS, Python allocator overhead,
caller-retained objects, or memory held by clients. A slot is not a trace and
does not explain model behavior.

`DeliverySlotRegistry` solves one narrow problem: a `TraceDelivery` may have
reserved ledger claims by the time its constructor returns. If a caller is
interrupted before assigning that returned object, a plain constructor call
can lose the only convenient handle. A registry retains the exact object
**before** calling its constructor. A trusted owner can retrieve the same ready
object using an identity recorded before construction. Each registry bounds
the number of active slots and the size of their identity strings; the
delivery and ledger impose their own separate limits. These are cooperative
logical accounting controls, not process-wide memory quotas.

## Owner sequence

1. The trusted owner creates `DeliverySlotRegistry(max_slots=N)` with a
   positive integer no greater than 65,536. It retains the registry itself in
   an independently recoverable process/job owner. An object with only a lost
   reference is not recoverable by this in-memory registry.
2. Before registration the owner records a bounded `slot_id` and a positive
   integer `generation` (less than 2^63). Generations are strictly increasing
   **across the whole registry**, including retired slots, not just per ID. The
   registry holds one high-water mark rather than an unbounded history. A full
   registry rejects additional slots. An interruption before insertion may
   consume the generation without creating a slot; retry with a larger one.
   Concurrent owners must serialize generation allocation with registration,
   or tolerate a higher generation registering first and retry the rejected
   lower generation with a fresh number.
   After insertion, `lookup(slot_id, generation)` recovers the exact slot even
   if the registration return is lost. The owner must give a new registry a
   distinct process/registry lifetime identity; the numerical generation
   alone does not disambiguate restarts.
3. `slot.construct(budget, store, **delivery_parameters)` uses the concrete
   `HostBudget`, `HostTraceStore`, and `TraceDelivery` types. It first installs
   the newly allocated child in the slot, then calls the delivery constructor
   outside the registry lock. There is no caller-supplied construction callback.
   The slot remains occupied while construction or cancellation is in flight.
   A failed construction before any child is installed, or with proven clean
   constructor cleanup, is marked `failed_clean` and can be retired. An
   uncertain interruption quarantines
   the slot and retains capacity. The owner must exclusively control the
   construction generation and not inject external claims into it: clean
   evidence only describes the delivery constructor's own work.
4. On success, `slot.adopt()` returns the same ready child on repeated calls.
   The owner can look it up again if the return-to-caller assignment is
   interrupted. Adoption is a recoverable reference handoff, not a transfer
   of exclusive ownership. The coordinator must prevent conflicting use and
   is responsible for the full producer, delivery, and consumer custody
   protocol in [TRACE_TRANSPORT.md](TRACE_TRANSPORT.md).
   The slot key selects a slot in this registry only: it does not identify
   or validate a job, sample, event, or the child's `DeliveryIdentity`. Before
   routing, the trusted coordinator must bind the slot to its registry/process
   epoch and expected broker/job/team/run/request/sample/row identity, then
   compare that binding with the adopted child's identity. Neither `lookup()`
   nor `adopt()` performs those checks or authenticates the caller.
5. `slot.cancel()` records cancellation under the registry lock. While the
   constructor is active, signaling is deferred until it returns; the slot
   cannot be adopted or retired in the meantime. For an initialized child,
   the actual `child.cancel()` call occurs outside the registry lock and may
   block on the child's signal lock or scheduling. A future broker must call
   it without holding its own control lock. Cancellation requested is not
   proof that work stopped, claims were released, or delivery drained; data
   already exposed under the delivery's own commit rules cannot be retracted.
6. Dispose sources and other aliases according to the delivery contract, then
   wait for `child.drain(timeout)`. `slot.retire()` never cancels and invokes
   `child.drain(0)` only to check that local disposal already completed; it
   returns false if not drained. An unconstructed idle slot and a slot with
   a proven clean construction failure can be retired directly. Retirement
   frees the registry slot, **not** arbitrary caller or consumer references.
   A stale capability cannot act on a reused ID, and a retired generation
   cannot be registered again in the same registry.

The constructor's private clean marker is set before any ledger request can
open, cleared before opening one, and set again only when guarded cleanup
finishes. A successful constructor leaves the marker false; successful
deliveries use the ordinary drain/retire protocol instead. A missing/dirty
marker or a failed cleanup is not
proof that a child is clean; the registry never infers this from ledger deltas.

If construction, cancellation, or handoff fails in an uncertain state, the
slot is quarantined and retains its child and capacity; `adopt()` and `retire()`
fail closed. `failure_guard` exposes a `DeliveryConstructionError`'s exact
ledger guard when available. Once the construction call has stopped,
`owned_delivery()` lets a trusted supervisor inspect the exact retained object.
This only checks slot construction/cancellation bookkeeping; it does not
establish that operations on a previously adopted child have stopped. That
object may be only partially initialized: its methods are unsafe until
the supervisor establishes otherwise. No API automatically scans or releases
claims. If an exception prevents even the guard from being recorded, do not
infer ownership from a ledger delta; quarantine/reconcile the affected process
under the surrounding supervisor. There is intentionally no public way to
retire a quarantined slot. Capacity can remain occupied until that registry is
replaced under supervised process-level recovery.

The implementation catches ordinary failures and tested interruptions around
construction and cancellation. Python cannot make arbitrary asynchronous
exceptions at every bytecode or lock/cleanup boundary atomic. A production
owner must retain and monitor its registry, reconcile interrupted operations,
and treat unknown states as quarantined rather than assuming cleanup. Registry
mutations and private fields, constructor parameters, producer ownership,
sink behavior, and custody attestations are all trusted. No broker or IPC job
may expose traces until the separate source, copy, result, serialization,
delivery, cancellation, and consumer ownership boundaries are wired and
independently validated.
