# In-process resident-worker broker

This is an initial local scheduler for one trusted resident worker. Many
threads can submit requests to the same broker; one owner thread executes
them serially on the same worker. It does not batch calls or establish
simultaneous GPU execution. Cross-process transport, GPU-wide coordination,
memory admission, and trace capture require additional components; none is
provided by this module alone.

`Broker(worker, max_pending=..., max_pending_per_team=...)` accepts a trusted
worker implementing `snapshot_request`, `execute`, `poisoned`, `close`, and
the exact `cancellation_exception` type. Submissions and worker snapshots
must have the exact `ForwardRequest` type; worker results must have the exact
`ForwardResult` type and contain a detached, mutable plain CPU tensor for logits
(inference-mode tensors are rejected because they cannot be mutated normally).
Both dataclasses must have exactly their declared instance fields: extra
attributes are rejected before the broker rewrites their request IDs.
Subclasses and trace-bearing wrappers are rejected. The resident worker
validates and copies bounded named CPU token tensors for forward inference;
arbitrary code, hooks, interventions, and tracing are not supported by this
contract. These checks prevent accidental use of an unsupported trace schema;
they do not sandbox trusted workers or prove tensor provenance. A trace-aware
path requires explicit ownership transfer, disposal, and delivery accounting.
Worker snapshots use only supported names, equal two-dimensional integer CPU
tensors, and a valid seed. The broker enforces absolute ceilings of 256 batch
rows, 8192 tokens per row, and 64 MiB across inputs; the worker must enforce its
own tighter model and workload bounds. If a worker returns any snapshot or
result that fails validation, its custody is unknown. The broker latches
`quarantined`, retains the object and worker, stops processing pending work,
and does not close the worker. A supervised owner must exit promptly with its
device lease and unsafe marker retained; starting another broker in the same
process is not recovery. Retention is best effort under catastrophic host-memory
failure and does not establish the fate of aliases outside this process. A
trusted producer remains responsible for retaining its own resources until it
has completed its forward-only handoff; the broker cannot make an arbitrary
Python return/assignment boundary atomic. A failed owner-thread start or worker
close also requires quarantine because its teardown state is unknown. For an
ordinary worker execution or snapshot exception, the trusted worker must prove
device completion and dispose its own resources before raising, or signal
quarantine itself. The broker does not infer those guarantees from the exception.
Raw host or PyTorch out-of-memory exceptions and process interruptions are
treated as uncertain and quarantined.
The caller must keep input data stable while it is being snapshotted. The
worker and model must not also be accessed directly by other code. This is
an in-process coordination mechanism for trusted clients, not a security or
process-isolation boundary.

```python
import torch
from looking_glass.broker import Broker, QueueFull, RequestState
from looking_glass.resident_worker import ForwardRequest, ResidentWorker

# Construct and configure a trusted evaluation model separately. Its parameters
# and buffers must already be on the worker's selected device.
worker = ResidentWorker('example', model.eval(), device='cpu',
                        max_batch_size=1, max_sequence_length=32)
with Broker(worker, max_pending=8, max_pending_per_team=2) as broker:
    request = ForwardRequest('caller-id', {'input_ids': torch.tensor([[1, 2, 3]])})
    try:
        handle = broker.submit('team-a', request)
    except QueueFull:
        # No request was accepted; the caller can retry under its own policy.
        raise
    outcome = handle.wait(timeout=10)
    if outcome.state == RequestState.SUCCEEDED:
        logits = outcome.result.logits  # bounded per request, owned by client
```

## Admission, identity, and dispatch

- `submit(team_id, request)` first reserves a slot, snapshots and validates the
  request, and then returns a handle. It raises `QueueFull` synchronously when
  the global or per-team pending capacity is full and `BrokerUnavailable`
  when shutdown or worker failure prevents admission. Invalid caller requests
  and ordinary worker snapshot exceptions propagate synchronously and release
  their reservation when accounting is known. A rejected worker-produced
  snapshot instead quarantines the owner and retains its reservation. A failed
  submission produces no handle or terminal outcome.
- `max_pending` and `max_pending_per_team` count preparation plus queued jobs;
  one additional job may be executing. Snapshot preparation can overlap other
  preparations and execution, but it is bounded by these slot counts. Slow
  preparation does not hold the broker lock and cannot block cancellation of
  another handle. The preparing request itself has no handle yet and cannot
  be cancelled through this API; a stuck snapshot can hold its slot and delay
  worker shutdown indefinitely.
- The broker also preallocates `max_pending + 1` job roots. Each preparation
  reserves one before invoking the worker, and a job stays rooted through
  queue publication, dispatch, cancellation and known cleanup. A full root
  table rejects admission, including when a removed job is still retiring.
  Uncertain queue removal or cleanup quarantines the owner and keeps these
  roots until process exit. The table bounds broker-owned job references, not
  total host memory or copies retained in caller, worker or cleanup frames.
  Retirement commits under the broker condition: quarantine that wins first
  keeps the input rooted; retirement that wins first completes its already
  authorized input disposal outside the condition, with its slot still charged
  until disposal returns. The worker must honor the forward-only ownership
  contract; a root cannot prove disposal of aliases owned by other components.
- Ready jobs dispatch in round-robin order among teams with queued work, FIFO
  within each team's **readiness** order. Concurrent same-team submissions can
  become ready out of submit-call order. This is fairness by dispatched job,
  not by GPU time or bytes, and does not preempt long calls. Immediate
  admission rejection does not provide fair waiting, priorities, or deadlines.
  Team IDs are caller-supplied strings, not authenticated identities; callers
  can create aliases. A service must authenticate and map callers separately.
- `handle.job_id` is a monotonically increasing identifier unique within the
  lifetime of one broker. Caller `request_id` values can repeat, so use
  `Outcome.broker_id` together with `job_id` for correlation across broker
  restarts. Each worker call receives a
  distinct internal request token, which is checked on return; the caller's
  original ID is restored in the result and outcome. This detects stale or
  misrouted results carrying a wrong token; it does not prove tensor
  provenance or protect against a dishonest worker.

## Completion and cancellation

Accepted jobs have `PENDING`, `RUNNING`, and one terminal state: `SUCCEEDED`,
`FAILED`, or `CANCELLED`. `handle.state` reads the current state. `wait(timeout)`
returns an `Outcome` for any terminal state and raises `TimeoutError` if the
wait expires; it does not cancel work. A handle's `cancel()` returns true
when it first requests cancellation, and false if already requested or
terminal. Cancelling a queued request is immediate. Running cancellation is
cooperative: the state remains `RUNNING` until the worker returns, and an
otherwise successful result is discarded. CUDA kernels cannot be preempted by
this flag. If execution instead fails, the failure is reported; a poisoned
worker's exception is not mislabeled as cancellation.

Terminal outcomes contain error type and message rather than an exception
traceback. Quarantined exceptions and their tracebacks are retained privately
until process exit because their frames may own unresolved resources; they can
also contain sensitive input references.
Public errors from a quarantined operation are fresh sanitized exceptions,
without a link to those private tracebacks. `KeyboardInterrupt`, `SystemExit`
and `GeneratorExit` keep their interruption categories; the worker must still
hold any opaque object that never reached the broker across a return boundary.
On a poisoned worker it fails pending jobs when cleanup is known. An
unexpected owner failure quarantines the broker and pending status may be
unknown. It rejects new work and does not automatically retry, lower precision, drop
measurements, or substitute a smaller workload. A result with a wrong internal
identity quarantines the owner and is retained. An outcome notification
can precede release of the owner thread's last local references, so it is not a
guarantee of immediate memory reclamation.
`broker.failed` remains true after a broker failure and differs from an
ordinary close; a service should check it when reporting its workload status.

If persistent host-memory exhaustion prevents queue cleanup or publication of
a terminal outcome, `wait()` and `cancel()` can instead raise
`BrokerUnavailable`: the request's status is unknown, not a completed failure.
The broker latches quarantine, rejects further work, retains unresolved
accounting, worker and request references where possible, and reports `BrokerShutdownError` rather than
claiming a clean shutdown. Catastrophic exhaustion can also prevent even this
best-effort notification; a supervising process must treat an unresponsive
owner as incomplete and preserve its resource quarantine. Once uncertainty is
latched, later worker poison or snapshot/owner failure does not drain pending
work or release unverified pending reservations.

`close(wait=True, timeout=None)` stops admission, cancels queued work, requests
cancellation of a running call, and waits for the owner to stop. It returns
false when it has not stopped by the timeout; another call can wait again.
`close(wait=False)` does not wait. The owner closes the worker only after any
running call and all preparations finish, unless quarantined. Worker-close
errors retain the exception and worker, quarantine the owner, and surface as
`BrokerShutdownError`; any resources a faulty close already released cannot be
recovered. A quarantined close reports failure promptly even when an orphaned
preparation or non-preemptible worker thread has not stopped; a supervisor must
terminate the owning process and preserve its resource quarantine. The context
manager waits without a timeout on exit only while ownership remains known.

## Timing and memory boundaries

Outcomes include `submitted_ns`, `ready_ns`, `started_ns`, and `finished_ns`
from `time.perf_counter_ns()`. Submitted is the start of the submit call; ready
is after validation and snapshot; started is dispatch; finished is terminal
publication. `started_ns` is `None` for work that never ran. Ready minus
submitted includes admission contention and input snapshotting; started minus
ready is ready-to-dispatch delay. These host timestamps are not CUDA event
measurements and do not establish kernel overlap. Compare clocks only when the
platform guarantees a common clock domain and record the environment.

The broker limits counts, not total bytes. The resident worker independently
bounds each input and output, and may make additional transient snapshots.
Results retained by clients can accumulate without a global host-memory bound.
The broker does not measure or reserve device memory, enforce trace-byte caps,
or control unrelated GPU processes. A service needs trusted, measured
model/workload resource bounds, live device-wide free-memory checks, bounded
result delivery, and a cross-process device lease before claiming those
protections. Treat prompts and returned tensors according to the deployment's
privacy requirements.

`WorkerSynchronizationError` means device completion could not be established.
Marking this broker unavailable or receiving a terminal outcome does **not**
prove GPU resources are safe to reuse. Quarantine the lease and GPU-owning
process and perform supervised process/device recovery before new work; do not
release accounting as if a drain succeeded. An isolated benchmark also needs
its own coordinated device window and synchronized timing protocol.
