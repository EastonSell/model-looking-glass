# Bounded team queue

`looking_glass.fair_queue.FairTeamQueue` is a thread-safe, nonblocking in-memory
queue for the pending-work stage of a broker. It does not execute jobs, manage
GPU memory, or cancel jobs that have already been dispatched.

```python
from looking_glass.fair_queue import FairTeamQueue, QueueEmpty, QueueFull

pending = FairTeamQueue(max_pending=128, max_pending_per_team=16)
pending.put('analysis', 101, {'prompt': 'example'})
pending.put('review', 102, {'prompt': 'another example'})
item = pending.pop()  # QueueItem(team_id='analysis', job_id=101, payload=...)
assert pending.cancel(102) is True
```

Both capacities must be positive integers (booleans are rejected). Each `put`
never waits for queue capacity and raises `QueueFull`, a `queue.Full` subclass,
when either limit is reached; it raises `ValueError` for an already-pending `job_id`.
Rejected puts do not alter the queue. `pop` removes and returns the next item,
or raises `QueueEmpty`, a `queue.Empty` subclass. `cancel(job_id)` atomically
removes a pending item and returns `True`; it returns `False` when the ID is no
longer pending. `len(pending)` counts pending jobs, not running jobs. If an ID
is not reused, cancellation and dispatch racing for the same job have exactly
one winner. A caller must separately handle cancellation and cleanup of running
jobs.

Team IDs and job IDs are hashable values with stable hash and equality while
pending; simple strings and integers are suitable. IDs are unique only while
pending. If an ID is reused, delayed cancellation of an earlier job can remove a
newer job with that ID. A broker should allocate IDs unique for its lifetime,
including completed and canceled job handles, and synchronize queue operations
with its job-state changes. Avoid custom
ID hash/equality methods that re-enter the queue or acquire the broker's locks.
An item is an immutable `QueueItem`, but its payload is held by reference and
may itself be mutable. A broker must validate or copy submitted payloads as
needed, and avoid payload finalizers that re-enter an external lock held while
calling queue methods.

## Dispatch order and limits

Within each team, pending jobs are FIFO. Teams with pending jobs are in a FIFO
rotation: a pop dispatches one job from the team at the front, then places the
team at the back if it still has pending jobs. A newly active team joins the
back. The last canceled job removes that team's rotation entry; a later job for
that team rejoins at the back. A team that has no pending job immediately after
its job is popped also leaves the rotation, even while that popped job is still
running. For example, if A's queue empties on dispatch and A then B submit during
A's execution, A's new job comes before B's.

A pending team's next turn cannot be overtaken by arrivals from newly active
teams. If its head job remains pending and dispatch continues, it is served
after the teams already ahead get at most one turn each; jobs earlier within the
same team must first receive their own turns. The global pending limit bounds the
number of active teams at any instant. This is a dispatch-order property under
continued service, not a wall-clock deadline: the queue does not preempt running
jobs or guarantee their completion.

Limits prevent one team from holding more than its own pending quota, but global
capacity can fill, and a team can repeatedly be rejected. There is no guarantee
of fair admission, fair GPU time, equal throughput, fair treatment of arbitrary
job sizes, small-job priority, aging-based priorities, or a latency bound. A
large job and a small job each take one dispatch turn. The limits count jobs,
not payload bytes, trace bytes, or device allocations. Resource admission and
trace storage caps must be enforced by the broker separately.

Methods take a private lock for each operation and do not wait for capacity or
work. For a fixed order of operations under that lock, the result is deterministic.
Racing threads can acquire the lock in different orders across runs, so the
queue does not promise reproducible ordering for concurrent submissions. It has
no background thread, and deliberately invokes no application callback. A
broker may hold its own condition lock around calls, provided its lock ordering
and payload/ID behavior avoid re-entry; it is responsible for notifications,
timeouts, and the lifetime of queued and running work.
