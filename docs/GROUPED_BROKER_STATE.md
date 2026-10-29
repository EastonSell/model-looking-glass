# Cohort launch and publication state

`CohortState` is a broker-private coordination primitive; it does not schedule,
invoke a model, verify sampled evidence, reserve host or device memory, or
authenticate clients. A future grouped broker must do those things separately.
The broker gives each member a unique service-generated job ID and worker
request ID and authenticates the owner. Only a trusted coordinator may see the
full cohort publication; individual clients must see their own result only.

The broker stages all members before one owner calls `try_commit_launch`. Its
condition lock linearizes that decision with cancellation. A cancellation
accepted before the decision blocks launch; `finish_cancelled_before_launch`
publishes cancelled status for those members and a no-launch failure for their
peers. Once the launch decision commits, a cancellation is a request to suppress
that member's successful output; the shared execution must still drain. The
decision records intent before the actual worker call, so `launch_committed`
does not prove that CUDA started. Only one dispatcher may advance or recover
the launch; another thread must not retry it after an ambiguous handoff.

The owner submits the exact plan-issued fully split `BatchOutcome` with ordered
job, owner, and worker request identities. The state binds an exact
`BatchProfile` and checks every result's shape, dtype and finiteness; logits
must also be ordinary detached, contiguous, separately backed CPU floating tensors
with exact storage sizes. It rejects overlapping output regions and unexpected
statuses. Finiteness checks allocate CPU scratch; the broker must hold a
conservative whole-batch host reservation that covers this work and all other
simultaneous buffers before calling publication. The broker must independently
bind the actual model and runtime, check sampled evidence and authenticate
that this exact outcome object came from the selected plan. String identity and
shape checks do not prove provenance. These checks are point-in-time checks:
the trusted plan, coordinator and all other holders must not write through any
tensor alias until custody is transferred for that individual output. The
receiving client may then mutate its own result under the broker's explicit
ownership contract. Cancellation and
publication are serialized: an accepted cancellation suppresses that member's
successful logits, while unrelated members receive their own outputs. A group
failure has terminal precedence over cancellation; every member is marked
failed and retains a `cancel_requested` flag. A post-launch failure must request
device quarantine even when all members had requested cancellation.

All member completions are constructed before **one shared publication pointer**
is committed. Handles must consult that shared record; writing each handle in
a loop does not preserve atomic fan-out under interruption. If construction or
result validation fails before commitment, the state becomes `unknown` and a
post-launch unknown requires quarantine. If notification or return is interrupted
after commitment, the exact publication remains recoverable. `wait` polls at a
bounded interval as well as responding to condition notifications, so a failed
notification alone does not strand waiters. An interrupted cancellation return
can be reconciled with `cancellation_requested`; an interrupted launch return
can be reconciled with `phase` under the single-dispatcher contract.

This is cooperative in-process coordination, not a security boundary. A later
host-custody or delivery failure needs an independent process-health latch even
if this state already published a success; `mark_unknown` cannot erase that
commit. CPython
can strand a lock if an asynchronously injected exception lands in a generated
lock-exit instruction; a supervisor must detect an unresponsive owner and use
process-level recovery. Do not inject asynchronous exceptions into ordinary
lock-holding or CUDA execution. `quarantine_required` reflects device risk; an
unknown prelaunch host-custody state may still require process quarantine.

The state never releases memory. The broker must retain the actual inputs,
launch, split outputs, exceptions/tracebacks, per-job claims and aggregate group
claim through every live reference. The state itself holds publication logits;
`retire()` merely drops this reference after the trusted broker attests that
all consumers, handles, callbacks, exception frames and other aliases have been
disposed. A returned publication can itself retain logits and must be dropped
before releasing claims. A notification or retirement exception can retain
frames as well. Quarantined or unknown states cannot be retired for reuse;
uncertain custody remains charged until the owning process is replaced.
