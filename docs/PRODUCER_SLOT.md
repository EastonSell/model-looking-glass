# Producer adoption slot (prototype)

`ProducerSlotRegistry` retains a trusted CPU `ProducerEnvelope` and the exact
request and per-stage claim guards across construction and adoption. It is a
standalone cooperative primitive, not a broker, scheduler, GPU path, or
physical host-memory limit. The legacy `ProducerEnvelope.acquire()` remains
available and does not itself retain its return object for a caller that loses
the assignment. Coordinated code should use a slot rather than assuming it can
recover a lost `acquire()` return from ledger claims alone.

The trusted coordinator records a bounded `slot_id` and a positive, globally
increasing generation **before** registration. `register()` returns no slot;
use `lookup(slot_id, generation)` to retrieve the retained capability even if
registration or caller assignment was interrupted. Generations are consumed
monotonically within one registry. Generation allocation and registration
must be serialized, or retried with a newer generation if a later one wins
first. Bind any external identity to a registry/process epoch: a generation is
not durable or unique across registry reconstruction, process restarts, or
different registries.

```python
from looking_glass.producer_slot import ProducerSlotRegistry

slots = ProducerSlotRegistry(max_slots=8)
slots.register("job-123", 1)          # caller records this pair first
slot = slots.lookup("job-123", 1)
slot.construct(budget, "request-123", profile)  # no returned envelope
envelope = slot.adopt()               # a lost assignment can be retried

# Use the trusted synchronous CPU forward. Dispose all buffers and aliases,
# then attest each producer stage as described in TRACE_PRODUCER.md.
for stage in ("input_copies", "output", "trace_bundle", "workspace"):
    envelope.confirm_disposed(stage)
envelope.close()
slot.retire()
```

`construct()` installs the exact envelope object before any request or claim
registration, and retains each guard before its respective mutation. A clean
admission failure releases only exact untransferred pre-forward claims and
closes its request; a failed or interrupted cleanup remains quarantined.
Other code must not adopt, use, transfer, or release the construction's guards
or claims until construction has stopped. This is a trusted exclusive-owner
precondition, not isolation against code with access to ledger internals.
Unrelated or transferred claims remain charged and prevent a clean close.

Cancellation blocks further adoption; a cancellation during construction is
remembered until the constructor finishes, then signals the envelope. It does
not release claims or stop a running forward. `owned_envelope()` and `guards()`
are for trusted supervision after the slot is quiescent. A quarantined child
may be only partially initialized; its methods are not necessarily safe. A
quarantined slot cannot be retired through this API. A supervisor must first
establish work completion, ownership, and disposal, using exact retained guards
or ledger recovery capabilities; process-level cleanup may be necessary.

After successful construction, disposal remains the envelope owner's
responsibility. `retire()` refuses a live or ambiguous producer and requires an
explicitly closed envelope; it does not itself dispose tensors or release
claims. The owner must synchronize publication against late cancellation and
meter every overlapping producer, delivery, store, serialization, and consumer
buffer separately. In particular a small tensor view can keep a larger external
backing allocation alive: use genuinely standalone storage or retain a separate
claim for its backing owner.

The registry is bounded by `max_slots`, but caller-owned objects and other
registries are outside that cap. The slot and ledger use Python `RLock`s.
Signal-delivered `KeyboardInterrupt` or other asynchronous exception injection
at a lock-exit bytecode can strand a lock and block in-process inspection; use
cooperative cancellation and a supervisor with process-level recovery for an
unresponsive worker. Process termination does not by itself establish disposal
of externally held buffers or claims.
