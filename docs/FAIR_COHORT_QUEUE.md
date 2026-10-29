# Atomic fair cohort selection

Status: the queue API is a CPU-only scheduling primitive. The current broker
does not use it and continues to execute one request at a time. It does not
run a model, acquire memory leases, authenticate a client, or certify that a
model is batch-separable.

A trusted broker may supply `compatibility_key=` when calling `put`. A key is
either `None`, which is not eligible for grouping, or an exact immutable
32-byte digest. The broker must derive the digest from its verified, canonical
model/checkpoint/adapter/runtime, input schema and shape, deterministic
execution settings, and independently measured separability evidence. Matching
keys only identify the caller's grouping label: they are not evidence of
compatibility or a security credential. Collision handling, correct key
construction, and checking the actual requests and active profile again
before launching are the trusted broker's responsibility. Use broker-owned
plain string team IDs and integer job IDs; the older generic queue API also
accepts arbitrary hashable objects whose custom methods may execute while the
lock is held. No callback or payload inspection is used to compare keys.

`peek_compatibility()` returns the current head key or raises `QueueEmpty`.
The value can become stale before the claim. `pop_compatible(key,
claim_id=..., allowed_sizes=(...), max_per_team=...)` atomically claims the longest prefix
whose **number of requests** is listed in the strictly increasing, bounded
`allowed_sizes`. It returns an empty tuple if the current head no longer
matches, if no approved size fits, or if the head has `None`. Empty queues
raise `QueueEmpty`. The caller must translate certified combined row counts
into request counts, accounting for each request's exact input row count.

The prefix follows the same per-team FIFO and round-robin order as repeated
`pop()` calls. Selection stops at the first incompatible head or when the next
team has reached `max_per_team`; it does not skip that team to fill a cohort.
New arrivals and cancellation cannot interleave with a claim. A caller can
dispatch the normal serial `pop()` path when grouping is unavailable; if it
waits for a larger group, it must impose a bounded wait and hold no queue lock
while waiting. The queue does not choose that delay, perform admission, or
decide which model profiles are valid.

Every attempt supplies a broker-generated `claim_id` known to the broker before
the call. A successful selection stores the full selected tuple in a bounded
queue-owned claim ledger as part of its state change. If an interrupt happens
after the queue commits but before the caller receives the return value, the
broker must call `get_claim(claim_id)` and recover or fail every selected
member; it must not assume an exception means no jobs were claimed. The broker
acknowledges with `acknowledge_claim(claim_id)` only after taking custody of all
selected job identities, input commitments, and cancellation state. Only then
does the queue discard its retained copy. The broker must use a fresh token for
every attempt throughout its lifetime, not merely while a claim is outstanding:
an interrupted acknowledgment can have committed even when it raised, and a
retry must never acknowledge a later claim with the same token. The queue only
checks that tokens are unique while outstanding; it cannot enforce lifetime
uniqueness. The number of outstanding claimed jobs is limited to the
configured pending-job capacity. When that ledger is full, selection raises
`QueueFull` without claiming more jobs. The pending count returned by `len`
does not include claimed jobs; the broker must account for both independently.
Pending and outstanding claimed job IDs cannot be duplicated in the queue;
the broker must also prevent reuse while dispatched work remains live. Once a
job is claimed, cancellation belongs to the broker's staged/running state,
and must not release the job's input custody early.

Selection uses a bounded shadow copy of queue bookkeeping and constructs the
result before committing. This may temporarily duplicate bookkeeping for the
pending and outstanding-claim limits; it does not duplicate payload storage.
The service must account for the metadata in its host budget. Interrupted
bookkeeping is rolled back when Python can perform the recovery, or the
completed claim remains recoverable by token. As with the existing queue,
abrupt process termination or persistent resource failure requires
broker-level recovery. The broker must retain identities and per-member
cancellation state until terminal publication or explicit unknown-status
recovery. It also needs a
single measured whole-batch device lease and host custody for snapshots,
concatenation, outputs, and copies before it can actually execute the group.
