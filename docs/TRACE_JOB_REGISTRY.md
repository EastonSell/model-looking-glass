# Recoverable trace-job receipts (prototype)

`TraceJobRegistry` is a bounded, in-process **control-plane primitive** for a
trusted owner. It does not run a model, reserve memory, own or store tensors,
validate trace coverage, authenticate clients, publish a manifest, or dispose
any resource. It is not connected to the existing broker, GPU service, or IPC.
Its phase changes are owner attestations; the future worker must actually
complete the actions and satisfy the custody rules described below.

A registry is bound to one owner-supplied 32-character hex broker lifetime ID
and one 32-character hex process-generation ID. The owner must establish that
those values identify the actual broker and execution process and use one
authoritative registry for that pair; this module cannot verify either fact
or coordinate separate registry instances. It admits at most `max_jobs` active job records. Job IDs are
positive bounded integers, with the broker-scoped internal worker request ID
`lg-{broker_id}-{job_id}`. A broker must not reuse an ID during that lifetime.
This also applies across replacement of a worker process: neither the process
generation nor registration ID is part of that worker-request string. It is
separate from a trace event's `(run_id, sequence)` identity; the trusted owner
must coordinate that event identity across all jobs and producers.
Before registration, the trusted caller also creates and retains a unique
32-character hex `registration_id` for that particular registration. It must
never reuse this ID for the same broker/job key. Recovery requires this ID, so
an old caller cannot recover a later generation that reused a numeric job ID.
Per-record generation and a private token similarly make previously issued
capabilities stale. These IDs are owner assertions, not authentication.

```python
import uuid
from looking_glass.trace_job_registry import TraceJobRegistry

jobs = TraceJobRegistry(broker_id="a" * 32, process_generation="b" * 32,
                        max_jobs=16)
registration_id = uuid.uuid4().hex      # persist with the job before registration
receipt = jobs.register(7, registration_id=registration_id)
receipt = jobs.recover(7, registration_id=registration_id)
# Trusted owner stages and validates the complete batch under separate claims.
jobs.seal(receipt)                       # owner attests that work completed
handoff = jobs.begin_handoff(receipt, "delivery-owner")
handoff = jobs.recover_handoff(receipt, "delivery-owner")
jobs.ack_handoff(handoff)                # trusted recipient accepts custody
# The responsible owner actually disposes all resources and aliases first.
jobs.begin_disposal(receipt)
jobs.finish_disposal(receipt)            # explicit owner attestation
```

Registration commits the bounded record before returning its receipt. If the
return is interrupted, the trusted owner can recover that record using the job
and registration IDs within the known broker and process generation. Similarly, a tentative
handoff is registered before its capability returns, so it can be recovered
for the same asserted recipient. A successful acknowledgement changes the
phase to `owned`; repeating it with the same live handoff is idempotent. A
stale, foreign or reused-generation capability cannot transition a new job.
Receipts and handoff objects reject ordinary copying and pickling, but Python
references can be shared and private fields can be accessed in this process.
Recovery operates on the registry chosen by the trusted caller; it does not
independently verify the actual broker or process. The registry is cooperative,
not a security or isolation boundary.

The phases are `preparing`, `sealed`, `handoff_pending`, `owned`,
`disposing`, and `quarantined`. `seal()` is only the owner's statement that
its entire batch is ready; it does not check events or publish values.
`begin_handoff()` does not itself transfer or release a memory claim, and the
sender must hold custody until acknowledgement and any required broker-owned
publication protocol finish. `begin_disposal()` changes metadata only.
`finish_disposal()` removes the record **only after** the owner attests that
all related operations stopped and every covered buffer and alias was
disposed or otherwise independently claimed. It cannot verify that assertion.
A `finish_disposal()` return can itself be interrupted. Recover with the exact
job and registration IDs: a still-live `disposing` record can be finished
after the owner re-establishes disposal; an absent or stale record means only
that this registration is no longer live. It is not independent evidence that
the resources or other owners' copies were disposed.
A quarantined record remains admitted and cannot be retired by this API;
trusted external reconciliation or process teardown is required.

`cancel()` first sets a cooperative signal outside the registry's publication
lock, then checks that the receipt is still current. Cancellation neither
stops work nor releases custody. Seal and handoff decisions inspect that signal
under the short lock: the final signal check is the decision point. If the
check wins the race, a transition can complete even when cancellation returns
before that transition's caller observes its return. An acknowledged handoff
remains owned after a later cancellation; the owner must route cancellation
and cleanup to the current custodian. No ledger call, model work, copy, wait,
or disposal occurs while the registry lock is held. Methods return their values
outside the lock to avoid one interrupted-return deadlock. Asynchronous
interruption of Python lock exit itself cannot be fully ruled out; if a lock
or transition becomes uncertain, quarantine and reconcile the owning process
instead of assuming cleanup succeeded. These cooperative rules are not a
promise of bounded wall-clock cancellation or atomic cleanup at every Python
interruption point.

Before enabling actual trace jobs, the worker must additionally:

* Bind authenticated job/team/input ownership to a stable exact CPU worker and
  hold input claims pinned while exclusively snapshotting the unchanged input.
* Admit the whole producer, metadata, per-event source and copy budget without
  fragmented partial reservations, and reserve a defensible upper bound before
  each allocation. Do not hold an admission lock across work or cleanup.
* Pre-register recoverable individual delivery slots and adopt exact handles;
  losing a constructor return must not silently orphan a claim.
* Validate every authorized sample, token, site and invocation, worker
  postcheck and run/sequence identity before sealing one complete batch. Keep
  per-event manifests private until that batch commit.
* Handle late cancellation, shutdown, transfer interruption, failed disposal,
  retained aliases, disconnect and release acknowledgement. Retain claims and
  quarantine when ownership or operation completion is uncertain.

This registry's CPU tests check only the control-plane transitions, bounds,
stale receipts and recovery from selected lost-return cases. They do not prove
aggregate host or GPU memory limits, broker handoff, batch publication,
cross-process custody or a causal explanation.
