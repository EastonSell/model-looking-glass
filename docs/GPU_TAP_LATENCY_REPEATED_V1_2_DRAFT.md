# Repeated tap benchmark v1.2 lean timing, prospective draft

This is a new, unexecuted protocol identity,
`gpu-tap-latency-repeated-v1.2`. It keeps the exact fixed input, checkpoint,
capture selection, six warmups, 36 measured samples, paired order and activity
screen from the v1/v1.1 proposal. It does not revise the earlier contaminated
pilot or permit pooling any v1.1 diagnostic with v1.2. The fixed supervised
owner is enabled for one provisional session; no v1.2 GPU measurement has yet
been reported. A multi-session aggregate requires separate coordination.

The previous diagnostic placed hundreds of expensive physical-owner checks
inside each tiny-model sample. A CPU-only supervised probe found this overhead
could dominate the intended comparison. For v1.2, the same exact owner,
physical lease and admission are checked before and after one bounded,
completion-synchronized workload. There are no repeated OS lease checks
between its internal model, hook, validation, event and synchronization calls.
The model result, logits, session, retained event container, capture and
observed stream wrappers remain rooted through the completion sync and the
outer postcheck. A failed operation or postcheck marks the owner unsafe,
retains known roots and claims, and permits no further CUDA calls before hard
process exit. A change or revocation during the workload may be detected only
at the postcheck; the supervisor and lease are cooperative point checks, not
atomic protection against mid-call loss or outside GPU pressure.

Each sample records three distinct intervals, with the same boundaries for
none, direct and gradient:

* `wall_ns` runs from immediately before the start-event record through the
  device-wide completion synchronization. It includes trace setup, forward,
  output/capture validation and stream checks. It excludes the later physical
  owner postcheck, checked disposal, second sync and allocator postcheck.
* `device_event_ms` spans the two recorded default-stream events. It covers a
  different interval from wall time and may include idle gaps while host work
  is performed. Neither measure is a pure kernel or hook-only cost.
* `guarded_wall_ns` runs from before pre-sample synchronization and memory
  preparation through the completion, owner postcheck, checked disposal,
  second sync and final memory postcheck. Its start and end are saved. It is
  an operational cost observation, not a correction to subtract from wall.

The source-stamped `completion_ns` closes the primary wall. `ended_ns` follows
the checked disposal and second synchronization, and the separately saved
postcheck time follows the final device-facing memory observations. The full
guarded interval must enclose all of them. After the final measured sample's
postcheck, there is no later device work until the parent completes its activity
screen and grants cleanup. Per-sample memory
and ownership checks, fixed host context, parent telemetry and disk
serialization can still affect the workload; their effects are not assumed
equal or absent. Record the actual cooperative CPU pauses and overlapping
work, and do not subtract estimated overhead from individual timings.

Preflight device preparation precedes the child ready frame. After the parent
grant, a fresh CUDA memory snapshot is taken under the enclosing owner guard,
with each scalar validated before the next query. Its source interval is saved
and must be at most two seconds old at the first model-call attempt. The
non-I/O start gate authorizes an interval no wider than 250 ms while preserving
the parent's shorter grant expiry and fresh activity bounds; the actual first
call stamp must lie inside it. The final owner/admission point check occurs
after the snapshot and gate and immediately before that stamp. This bounds
observation age but cannot eliminate a scheduler pause or an external memory
change after any check.

The clock-resolution screen, event/wall consistency check, postcheck ordering,
source/identity binding, parent activity barriers, durable journal, retained
failure evidence and strict reaped-owner requirements remain mandatory. The
first successful fresh owner can be reported as one provisional session; it
is not a five-session aggregate or a claim about general performance.
