# Private trace-batch custody prototype

Status: **trusted, process-local CPU prototype**. It validates and privately
stores a complete single-job GPT-2 batch, then commits one metadata manifest.
The batch primitive does not run a model or expose a sending API. The separate
[CPU trace owner](CPU_TRACE_OWNER.md) composes it with a trusted CPU traced
worker and producer envelope. Neither is connected to the broker, GPU, IPC or
network. They do not authenticate callers or prove that values came from a
given model or input. Trace values do not by themselves explain model behavior.

`TraceBatchRegistry` keeps a bounded number of private job contexts, including
contexts still being constructed. A trusted owner supplies the exact
`HostBudget` and `HostTraceStore`, broker lifetime and process-generation IDs,
a `BoundTracePlan`, and actual shape and model facts. It records a fresh
registration ID and a **strictly increasing registry-wide generation before
calling register**. `lookup(job_id, registration_id=..., generation=...)`
recovers the same private owner after a lost return; obsolete generations are
rejected even if a numeric job ID or registration ID is repeated. After an
interrupted registration, first look up the exact key: the job may already be
installed. A pending registration is listed in `active_jobs()` but cannot yet
be looked up; do not assume it is absent while its attempt may still be live.
If it is absent and the original attempt is known to have stopped, retry with
a higher generation; an attempt may have consumed its old one.
The caller must serialize generation assignment and keep the registry itself
rooted in a known supervisor/process epoch. As with the broker, numeric job IDs
must not be reused across a broker lifetime, including process replacements;
the registry's local generation does not make worker request IDs globally
unique. Broker ID, process generation, job ID, registration ID, generation,
and the selected registry are trusted owner assertions. A new registry can
reuse numeric generations, so its process/registry epoch must be distinct.
The worker request ID `lg-{broker_id}-{job_id}` includes neither process nor
registration generation. The owner must separately allocate unique
`(run_id, sequence)` keys and correct per-run forward indices across jobs and
producers.

Before construction, callers must stop changing plan objects. The registry
reconstructs and validates a private, bounded `TraceJobLayout`; its limits are
trusted configuration, not proof that the input or model matches those facts.
The layout currently covers one job owning every row of one physical batch,
one invocation per selected GPT-2 site, CPU float32 snapshots, and output-local
token coordinates. It rejects missing, extra, duplicated, misordered or
misidentified events and overlapping reported storage spans. It does not prove
the source allocations have no external backing owners. See
[TRACE_JOB_VALIDATION.md](TRACE_JOB_VALIDATION.md) for its detailed contract.

## Owner sequence and limits

1. The owner makes a separate defensible reservation for the producer's
   input, output, complete source bundle, intermediate workspace, and any other
   participating live buffers. It exclusively controls the unchanged input,
   model and producer outputs. This module does not make or verify those
   reservations, run a forward, hold the model lease, or establish device
   admission. It never co-batches independent client jobs.
2. After `register`, recover the job by its exact key and call `prepare()`.
   It roots all child slot IDs/generations inside the job before construction,
   then reserves each child's 65,536-byte metadata envelope and source claim
   before capture. These source claims overlap intentionally with the producer
   bundle claim. No capture may begin until `prepare` has successfully
   completed and the owner has independently checked the real model/input.
   Registration and preparation must not race with mutation of trusted plans or
   capabilities. `prepare` can fail after partial admission; claims stay rooted
   and require the explicit cleanup sequence below. If a source acquisition's
   result is interrupted before the owner records whether it succeeded, the
   pending state is uncertain and requires supervisor reconciliation; it must
   not be guessed absent. An interruption at a successful return may leave the
   job already `prepared`; recover and inspect the exact rooted job rather than
   assuming the call made no change.
3. On successful capture, retain exclusive custody of the complete source and
   call `stage(events)`. It validates the whole set first, then maps each event
   to `(row, layer, module kind)` and privately stages its own child copy.
   Copies allocate new store claims as needed, so capacity contention can
   reject a job mid-stage. This is a nonblocking fail-and-owner-cleanup path,
   not atomic whole-bundle admission or a promise of capacity. Never wait for
   more capacity while indefinitely holding a partial bundle; an external
   scheduler must arrange fair retry.
4. Once staging has stopped, the owner drops the full forward result, event
   tuple, loop variables, temporary maps, traceback frames and **every other
   source alias**, or accounts for them separately. It must then call
   `confirm_sources_disposed()` as an explicit truthful attestation. Error
   cleanup is best effort, and an exception group, custom exception attribute,
   external tensor backing owner, or another caller can still retain a source.
   A failed validation or a returned exception is never proof of disposal.
   If an operation is still active, this method rejects the attestation. After
   cancellation before capture, the trusted owner may attest that no sources
   were created; for an uncertain acquisition outcome it must quarantine.
5. After confirming source disposal and completing all trusted model, input,
   batch, and other postchecks, call `seal()`. It keeps individual child
   manifests private while verifying the exact child delivery identity and
   full validated event metadata, including site, row, token order, forward,
   invocation and sequence. It rejects repeated child delivery IDs in the
   batch. The final cancellation-signal check under the short job lock is the
   commit decision. `manifest()` then returns a fresh, tensor-free complete
   manifest. That method's successful read is decided by its liveness check
   under the job lock; a read already authorized there may finish after a
   concurrent cancellation returns. A later broker must apply its own
   publication/late-cancellation decision and must never infer that a returned
   manifest is a sent or complete consumer payload. An interruption after the
   seal commit can likewise leave a recoverable `sealed` job; inspect its state
   and apply the owner's publication and cancellation decision rather than
   repeating work or assuming no commit.
6. To abandon or finish the private job, call `cancel()`, truthfully confirm
   source disposal if not already done, then call `dispose(timeout=...)` and
   finally `registry.retire(job)` only after disposal returns true.
   Cancellation sets its signal first and then briefly synchronizes with phase
   commits; no child, store, model, ledger operation or cleanup is run while
   the job publication lock is held. Cancellation does not stop a copy or
   release memory. `dispose` signals children without the job lock and waits
   for their local drain; `timeout` bounds the drain waits, not time blocked
   inside a child cancellation call. Run cleanup off latency-sensitive threads.
   A timeout, uncertain transition, or quarantined child retains the job and its
   relevant claims for supervised reconciliation. Arbitrary asynchronous
   Python interruption cannot be made atomic at every lock/cleanup boundary.

Each child reserves its own source, store and eventual serialization claims.
An umbrella reservation cannot simply be held on the same budget: the child
would request a fresh claim and double-count it. Whole-bundle guaranteed
admission needs an allowance honored by every claim path or transferable
pre-reserved claims. This prototype's stage failure instead requires rollback
and fair retry after real cleanup. The ledger limits declared logical bytes and
items for cooperating users; it does not bound allocator overhead, model
parameters, Python objects, arbitrary external buffers, process RSS, clients or
system-wide memory. The returned manifest, its additional metadata copies and
temporary JSON reconstruction workspace are caller-owned and outside the
ledger; a future delivery owner must reserve and retain those separately.

The child deliveries never leave the private owner through this API, and no
payload or serialization capability is exported. The underlying child UUID is
probabilistic and historical uniqueness is not enforced. Any future ACK must
bind authenticated process, job, registration and slot generations as well as
the exact child identity, with explicit custody transfer and disposal on late
cancellation, failed publication, disconnect and shutdown. None of those wire
or consumer guarantees is provided here.
