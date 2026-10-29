# Producer host-custody envelope (prototype)

`looking_glass.trace_producer` provides cooperative admission and custody for
one **trusted synchronous CPU forward**. It is a standalone building block:
the broker, traced worker, trace store, transport and IPC do not currently use
it. It does not enable shared-model tracing or GPU execution.

A trusted coordinator supplies a model- and workload-specific `ProducerProfile`
with four `ClaimLimit(nbytes, items)` commitments. Profile versioning and
binding to a specific workload are external responsibilities and are not
implemented by this helper. The
commitments must conservatively cover every simultaneous allocation the owner
attributes to each stage:

* `input_copies`: worker-owned input snapshots, including both copies if they
  coexist. Caller- and broker-owned inputs require separate claims.
* `output`: the returned CPU output, such as logits. Any simultaneous model
  output or conversion buffers must also be included in the profile.
* `trace_bundle`: the entire returned bundle of captured source tensors,
  including the maximum number of sites, invocations, rows and tokens admitted
  by the trusted capture plan. The bundle claim is indivisible.
* `workspace`: forward temporaries, model intermediates, metadata, conversion
  and runtime overhead according to the coordinator's conservative estimate.

`ProducerEnvelope.acquire(ledger, request_id, profile)` reserves all four
claims before the callback can run. An unsuccessful admission attempts to
roll back every acquired claim; if interrupted during registration or cleanup,
some empty or claimed ledger records may remain and a trusted coordinator can
inspect `active_requests()`, `active_claims()` and `active_operations()` for
recovery. The coordinator must establish actual ownership and completion
before releasing anything. IDs have the same bounds and uniqueness rules as
`HostBudget.open_request`. Admission uses the public request guard to retain
the exact generation across registration and adoption interruptions. If
cleanup cannot finish, registered guards and outstanding ledger handles remain
discoverable to the trusted coordinator. Failed admission cleans up only
untransferred claims with that acquisition's unique producer owner label,
matching the exact request generation and producer stage; unrelated or
transferred claims remain charged and keep the request open. Owner labels are
visible metadata, not authentication: cooperating code must not reuse another
acquisition's label.

The four reservations are sequential, not one atomic bundle admission. Two
concurrent acquisitions can each reserve part of the available capacity and
then both fail even when one complete profile would fit. The coordinator must
serialize or atomically schedule whole-bundle admission and fairly retry failed
requests; this helper does not provide a fairness policy or queue.

`envelope.run(forward)` pins all four claims as one ledger operation, invokes
the callable once, and finishes the operation before returning a result. It
requires synchronous work that has completed before return. It rejects declared
async and generator functions. If a callable returns a recognized future,
awaitable or generator, the envelope tombstones the request and quarantines
the object and operation, keeping the claims pinned. A quarantined envelope
does not permit disposal or close, even if someone externally unpins its
operation, because the envelope may still own the deferred object and its
result. There is no automatic quarantine recovery. A trusted coordinator must
first establish that work has stopped, dispose the envelope and its deferred
object as well as every other relevant reference, and only then use ledger
recovery capabilities to finish operations and release claims. The helper
cannot detect arbitrary background threads, async work hidden inside
containers, or calls that bypass the ledger.

Cancellation is cooperative. `cancel()` blocks new work and tombstones the
request but does not release claims or stop an already running callback. A
cancelled in-flight forward discards its result once the callback exits; the
claims remain until their owner attests disposal. Interrupted starts or finishes
are quarantined where completion is uncertain. Callback failures likewise
retain the claims, including failures whose traceback or exception payload may
still hold references. The helper strips the callback traceback before
re-raising, but cannot guarantee that other aliases or exception objects have
been destroyed. A failure is treated as an exited synchronous callback only if
its traceback contains that callable's own Python frame. Other failures,
including an interruption between return and result assignment or an exception
without an inspectable callable frame, are quarantined because completion is
not established. If cancellation itself is interrupted, retry `cancel()`;
the ledger tombstone is set before the envelope reports local cancellation.

The cancellation decision is serialized with the envelope's operation finish.
Cancellation that occurs after this check can race with returning the result.
The coordinator must therefore serialize its own final publication and
cancellation decision; the helper alone does not guarantee that a result will
never be published after a late cancellation.

`confirm_disposed(stage)` is a **trusted attestation**, not evidence of
disposal. Call it only when every buffer and alias covered by that stage has
been destroyed, or a separate claim already covers every live transferred
buffer. The ledger rejects release while an operation still pins the claim.
`close()` succeeds only after every claim and operation is resolved. A leaked
result or unhandled exception must not trigger automatic release.

For the trace-delivery prototype, reserve a distinct delivery-source claim for
each fixed capture slot **before** the forward, against the same ledger, in
addition to the producer bundle claim. Also reserve every simultaneous store
copy, serialization buffer and consumer-held payload separately before it is
created. Keep the producer bundle charged until the producer and all of its
bundle aliases are gone; release a delivery source only after its own source
aliases are gone. The store's own quota and the producer claim do not provide
the consumer or transport claims. This conservative duplication consumes
capacity even if it counts one buffer twice during handoff; it avoids an
unaccounted transfer between separate request generations. Unused slots must
be cancelled and disposed explicitly. Publication must make metadata and
payload visible atomically after worker postchecks.

These are declared logical commitments, **not a physical host-memory limit**.
Python, PyTorch, allocator rounding, retained framework state, external
references and system activity can exceed the estimates. Neither the ledger nor
this helper intercepts allocations or verifies a profile against the actual
workload. A trustworthy integration must bind a validated profile to an
authorized fixed CPU plan and account for every overlapping copy through its
last consumer acknowledgment; unsupported or unmetered paths must remain
disabled.
