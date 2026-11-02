# Cooperative GPU-owner supervision

`looking_glass.supervisor` coordinates the lifetime of one trusted,
directly spawned GPU-owner process. It creates a sticky unsafe marker before
the child can initialize CUDA, requires the owner to hold the shared device
lease, and clears the marker only after a matching clean-shutdown
attestation, confirmed child exit/reap, and reacquisition of that lease.

This module is a building block, not a complete GPU service or an isolation
boundary. All cooperating broker and benchmark launches must use this same
supervisor protocol and one agreed, trusted physical-device identity and
workspace root. It does not stop other programs from using the GPU, prevent a
trusted child from lying about its cleanup, or prove CUDA safety on its own.

## Identity and one shared state location

Use the same canonical absolute device lease path and marker path for all
owners of the physical GPU. The marker is derived from a `.lock` path:
`GPU-<uuid>.lock` maps to `GPU-<uuid>.unsafe.json`. The controller lock
and attestation use the marker path with `.controller.lock` and
`.attestation` appended. The directories must already exist on a trusted
local filesystem. Never unlink or replace the persistent lock files.

The supervisor checks this filename pairing and rejects hard-linked lock
files, but a different workspace root or a different lock name can still form
a separate namespace. For GPU work, callers **must** supply the reviewed
physical-UUID `DeviceIdentity` as `identity=`; the parent checks its full
path bundle while holding the device lease, and the child re-resolves and
rechecks the same bundle before CUDA access. Identity-less operation is only
for CPU fakes and tests and masks `CUDA_VISIBLE_DEVICES` in the child; GPU
entrypoints must reject missing identity before CUDA. This environment mask
is not a sandbox against a process that deliberately changes it. A
trusted external inventory and one common configured root are still required.
The child must compare the actual CUDA runtime UUID with that identity before
loading a model. A numeric CUDA ordinal or GPU name is not sufficient
evidence. Separate roots and nonparticipating processes do not coordinate.

Every broker, benchmark, and other participating GPU owner must launch
through `run_supervised`; merely taking the device lease and inspecting the
marker omits the controller lock and is not equivalent. Controller lock,
device lock, marker, and attestation belong to the same protocol. A remaining
marker or attestation, a malformed file, an incorrect identity, and a failed
or uncertain cleanup all block a later supervised owner. Do not automatically
delete quarantine state or infer safe recovery from the device lock becoming
free.

## Launch and child contract

```python
from looking_glass.supervisor import run_supervised

# Obtain these canonical paths from the shared, verified device identity.
result = run_supervised(
    direct_owner_command,
    device_lease_path=identity.lock_path,
    marker_path=identity.marker_path,
    identity=identity,               # required for GPU owners
    timeout=600,
    lease_timeout=30,
)

# A safe teardown is not the same as successful research output.
if not result.clean:
    raise RuntimeError(f"device remains quarantined: {result.reason}")
if result.exit_code != 0:
    raise RuntimeError(f"owner workload failed: exit {result.exit_code}")
```

`run_supervised` first takes the marker's controller lock and then the
device lease, checks that no state remains, and creates and flushes an
exclusive active marker with a fresh run ID. It releases the device lease for
the directly spawned child while retaining the controller lock. The command
is an argument sequence; no shell is used. The child receives its run
identity and configured paths in its environment.

Before starting the child, the supervisor roots a bounded, process-local
recovery record and prepares a private one-shot launch pipe. With the standard
`subprocess.Popen` class, the record also owns the Popen object before its
constructor runs. The parent grants startup only after that record holds a
usable exact direct-child capability. `OwnerSession.from_environment()`
checks the parent and waits for the grant before returning; every trusted
owner entrypoint must call it before acquiring the device lease, importing
CUDA-capable dependencies, or using CUDA. An ungranted owner rejects EOF and
exits after a 30-second gate deadline even if a parent-side writer is leaked.
That deadline starts only once the child enters the gate. This is a
cooperative protocol: arbitrary child code, injected Python modules, and
nonparticipating applications can bypass it.

The grant and subsequent wait are covered by cleanup using the recorded exact
child. If termination or waiting fails or is interrupted, the marker and the
recovery record remain; a still-live parent process is not evidence that its
supervisor thread remains active. Once the launcher thread is known quiescent,
trusted in-process recovery can enumerate `pending_supervised_children()` and
call `record.recover(terminate_grace=...)`. It operates only on the recorded
Popen capability and never clears a marker. An unattempted record can be
retired without a child; an attempted spawn with no usable process handle is
unknown and cannot be reported as reaped. In particular, interruption inside
the OS spawn before Popen stores its handle may leave an unidentifiable child.
No grant is sent in that case, and the marker remains for operator/process
recovery; do not guess a PID or use lease reacquisition as proof of exit.
The registry holds at most 64 records and fails closed when full. Its
capabilities are process-local and recovery requires external coordination;
they are not an authentication facility. By default this protocol tracks the
direct child only. The optional Windows process-job contract below adds a
separately supplied, cooperative process-tree check.
The `run_supervised(timeout=...)` child-wait deadline starts after spawn and
grant return; it is not a hard deadline on a stalled native spawn or grant
write. A stuck launcher may require process-level recovery with the marker
left intact.

For a startup channel, launch authorization and channel failure/cancellation
are ordered under a short lock that contains no I/O. If cancellation wins,
no grant is sent. If authorization wins, cancellation is post-authorization:
the grant can still reach the child after `abort()` returns, and the supervisor
must stop and reap the recorded child or retain its custody and marker. Abort
does not wait for a stalled spawn or grant write. The same ordering applies to
reader failure and receive timeout. If asynchronous interruption makes a
writer-descriptor close uncertain, its ownership is forgotten before close to
avoid closing a later, unrelated descriptor; an interrupted close can leak
the old descriptor and require process-level recovery.

`SUPERVISOR_LAUNCH_CUSTODY_CONTRACT` is the compatibility marker
`gated-rooted-child-v1` for these cooperative semantics, including
the unknown-spawn limitation. It is not an independent security proof.
On Windows, cooperating spawns serialize the interval in which the pipe
handles are made inheritable and pass an explicit handle list. Other
broad-inheritance spawns in the same launcher must be excluded. The general
CPython asynchronous-interruption limitation also applies to Python lock and
descriptor cleanup: a stranded lock or an uncertain descriptor may require a
process supervisor to restart the launcher while retaining quarantine.

By default, `env=` overlays the parent environment for compatibility. GPU
launchers that prepare an explicit trusted environment must instead pass
`inherit_environment=False, env=prepared_environment`; the child then receives
only that mapping and supervisor-owned fields. Include any OS variables needed
by the child explicitly. This prevents omitted parent values such as
`PYTHONPATH` from being reintroduced before the interpreter starts. This option
does not authenticate or sanitize variables deliberately included in the
mapping. Reserved supervisor fields are stripped case-insensitively before the
run's authoritative values are added; identity-less CPU mode also replaces any
case spelling of `CUDA_VISIBLE_DEVICES` with an empty mask. The parent process
environment is not modified.

### Optional Windows process-job custody

A dedicated Windows launcher may pass `process_job=job` to `run_supervised`.
The caller must provide a fresh, empty, independently reviewed job capability
and must not share it with another run or assign unrelated processes to it.
The reviewed `WindowsJob` in `scripts/v2_source_guards.py` supplies this
interface. The trusted capability must provide `assign(child)`,
`process_counts()`, `terminate_and_wait(timeout=...)`, and `close()`.
Its counts must be a pair of exact nonnegative lifetime and active integers;
both must initially be zero. Its drain method must
establish complete termination and handle custody for the job's members.
For this option, `terminate_grace` must be greater than zero and at most ten
seconds, matching the reviewed job implementation's drain limit.
The supervisor cannot authenticate the implementation or detect a capability
that lies. A caller that uses another implementation must independently verify
those properties.

The supervisor registers the job with the rooted child custody record, assigns
the exact child before the launch grant, and prevents concurrent reuse of the
same job object. Once the exact direct child is reaped, the supervisor requires
an observed zero active job-member count, then confirms the job drain and
closes its handle before considering marker removal. If members were still
active, it stops and drains them but leaves the marker unsafe. An unknown,
malformed or failed count, drain or close also leaves the marker and retains
the child/job recovery record. No failed drain is automatically retried.
Once the launcher is known quiescent, explicit `record.recover(...)` may retry
the supplied capability's recovery protocol; it never clears the marker.
If validation or registration fails before the custody record is registered,
the supervisor does not drain or close the supplied job. The caller remains
responsible for closing a fresh, unassigned job. If a caller erroneously passes
a job already held by another custody record, it must not close or otherwise
touch that other run's job.
If cleanup receives `KeyboardInterrupt` or `SystemExit` while handling an
ordinary launch error, the interruption is raised with that error as its cause.
An earlier fatal interruption retains precedence. The no-job path does not
require a process-job capability.

The supplied `WindowsJob` checks its lifetime accounting against retained
process handles. A short-lived member that escaped observation makes that
inventory unconfirmed even if the active count later becomes zero. The owner
must prevent or account for such descendants. The contract assumes the job
handle is not leaked and no external process can assign or break away from the
job; it cannot prove the absence of escaped or otherwise nonparticipating
processes. It does not itself verify CUDA teardown. The owner must still stop
all work and unsafe-reporting threads before attesting, and the launcher must
reconcile child exit, job evidence, and device state. Spawn, assignment and job
drain can stall outside the child-wait timeout; use an external deadline and
retain quarantine if the launcher is uncertain.

```python
from looking_glass.device_lease import DeviceLease
from looking_glass.supervisor import OwnerSession

session = OwnerSession.from_environment()
lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
session.check(lease)                 # before CUDA initialization

# The supervisor supplied a physical-UUID identity; reject a missing identity.
if session.identity is None:
    raise RuntimeError("GPU owner requires an identity-aware supervisor")
# Obtain runtime_uuid from the CUDA runtime, after the lease/marker check.
session.verify_runtime_binding(
    lease, runtime_uuid=runtime_uuid, device_count=1, cuda_device_index=0,
)
# Only now load a model and use this verified device.
# ... run the service or benchmark ...

# Stop admission; finish/cancel all jobs and preparations; join broker threads;
# verify all memory/resource claims are released; synchronize the device;
# recheck the worker's quarantine latch and runtime device identity. Only if
# each of these steps is known to have succeeded:
session.attest_clean(lease)          # immediately before exit
# Retain lease and do not launch any more GPU work; exit the owner process.
```

`check` requires the actual live `DeviceLease`, validates the active
marker and supplied physical identity, requires the caller to be the
supervisor's direct child, and starts a process-lifetime watchdog before
returning. A session binds to its first checked lease object: releasing and
reacquiring a different lease cannot reuse an earlier runtime verification.
In identity-aware mode, `attest_clean` refuses to succeed unless
`verify_runtime_binding` has checked the caller-supplied runtime UUID and
device count against that identity. The UUID must come from the actual CUDA
runtime, not from the preflight inventory. On Windows the watchdog waits
on a supervisor process handle and verifies its creation time to reject PID
reuse. On POSIX it observes parent reparenting. If the supervising process
dies or the watchdog cannot establish liveness, the owner exits without a
clean attestation; the marker stays. The watcher is best effort, not a
hard-real-time deadline, and cannot override a process stuck in uninterruptible
native code. Repeated checks reuse the same watcher.

The direct child must itself be the CUDA owner; do not launch the owner via a
wrapper or create CUDA-owning descendants. On this Windows machine, the shared
venv's `sys.executable` is a launcher that starts a different underlying
Python PID. Launch `sys._base_executable` directly and explicitly provide
the required venv site-packages and repository paths through `PYTHONPATH`,
or use another verified direct executable. A wrapper-launched descendant
fails the parent check before device work.

`attest_clean` is an assertion by the trusted owner, not an independent
measurement of CUDA. It must be the last action before exit, after a verified
drain, no live resource claims, successful synchronization, and a final
quarantine check. The owner must not release its device lease before process
exit. If any check is uncertain or fails, exit without attesting; never mark a
device safe just because a client disconnected or a timeout elapsed.

Run CPU-only cache, manifest, and input validation before starting a GPU owner.
Once an identity-aware owner starts, there is no unchecked "unused device"
attestation: even an early failure leaves the marker unless the owner
establishes the actual runtime binding and a clean drain. This avoids claiming
that an implicit or partially failed CUDA initialization never happened.

## Completion, failure, and recovery

The parent waits for and reaps the exact child process. It then reacquires the
device lease and verifies the run ID, child PID, marker, and attestation. Only
after all checks pass does it remove the attestation and marker. The returned
`SupervisionResult.clean` describes device teardown alone; `exit_code`
preserves the owner's separate workload status. A nonzero workload exit can
therefore coexist with `clean=True` when cleanup was demonstrably complete,
but it must never be presented as a successful experiment.

A timeout triggers termination of the exact supervised child and, after a
bounded grace period, a kill if needed. Timeout, a forced exit, missing or
mismatched attestation, or inability to establish cleanup leaves the marker.
The supervisor never searches for or kills unrelated processes. If the
supervisor itself dies, the child watchdog exits the owner without attesting,
and the marker remains; a later entrant refuses automatic reuse. Failed
process creation also leaves the marker conservatively in place.

If the marker is unexpectedly lost while the controller is alive, the
supervisor attempts to restore it before releasing its controller lock.
A filesystem failure that prevents restoring it keeps that lock held in the
current process and requires external intervention. This protocol assumes the
state directory is protected from unrelated modification. If the storage or
machine is unreliable, do not infer that a missing file establishes safety.

## Private startup delivery

For a trusted service that must return an address and short-lived credentials
to its trusted launching process, `StartupChannel` provides one anonymous
child-to-parent frame. The caller chooses `max_bytes` from 1 to 65,536; the
transport accepts one nonempty `bytes` value and bounds the announced length
before allocating its body. It does not use pickle, print the payload, or save
it in the marker, command arguments, or environment. The inherited endpoint
number and size bound are not credentials. The caller must validate the
decoded schema and arrange separate trusted delivery to each recipient.

```python
from concurrent.futures import ThreadPoolExecutor
import os
from looking_glass.supervisor import StartupChannel, run_supervised

channel = StartupChannel(max_bytes=65536)
pool = ThreadPoolExecutor(max_workers=1)
owner = pool.submit(
    run_supervised, direct_owner_command,
    device_lease_path=identity.lock_path, marker_path=identity.marker_path,
    identity=identity, startup_channel=channel, timeout=600,
)
try:
    message = channel.receive(timeout=30)
    if not channel.wait_closed(timeout=5):
        raise RuntimeError("startup reader teardown is uncertain")
    # Decode and validate message, then deliver team-specific credentials
    # over a separately protected channel. Do not print or persist message.
except BaseException:
    channel.abort()  # also use this if schema validation rejects a frame
    try:
        owner.result(timeout=20)
    except BaseException:
        os._exit(70)      # uncertain launch: do not reuse this dedicated launcher
    if not channel.wait_closed(timeout=5):
        os._exit(70)      # a retained writer left a reader pending
    pool.shutdown(wait=True)
    raise
# After requesting service shutdown through an independent control channel,
# wait for owner.result(), require clean=True and exit_code == 0, and close
# the executor. A delivered frame does not establish a successful workload.
```

In the validated direct owner, after the device binding and after the service
is ready, call `session.send_startup(lease, message_bytes)`. It checks the
same live lease and, for GPU identity mode, requires the runtime binding to
have succeeded. The endpoint becomes non-inheritable when
`OwnerSession.from_environment()` takes ownership of it. The sender can use
the channel once and then its write endpoint is closed. Do not send until the
server is actually ready. The parent must call `receive` with a finite deadline
and call `abort` if the application rejects the returned bytes; on a reported
channel failure the supervisor terminates only the exact child and retains the
unsafe marker when it observes failure before completing clean teardown.
If a fully attested child has already exited and its marker was cleared before
the application rejects the frame, the rejected startup still fails, but a
later `abort()` cannot undo that completed device teardown. `close()` rejects
an unused or unreceived frame; after a successful receive, use `abort()`
explicitly to reject it.

The launcher must be a dedicated, trusted process with no concurrent process
creation that broadly inherits handles. On Windows, Python must temporarily
mark the write handle inheritable even with `STARTUPINFO`'s explicit
`handle_list`; this module serializes its own spawns, but cannot prevent an
unrelated `close_fds=False` spawn from inheriting that handle during the brief
window. On POSIX it uses `pass_fds`. A hostile same-user process, a malicious
owner or launcher, and independently exposed credentials are outside this
cooperative contract. Never claim this is a security sandbox.

The synchronous pipe is read by a daemon thread so `receive` can return on a
deadline without waiting for a blocking pipe operation. The shared partial
buffer is cleared on cancellation, but erasure is best effort: Python and the
OS may retain immutable or allocator copies. If another process holds a write
endpoint, that reader can remain blocked after timeout and consume a reader
slot; at most 16 startup readers can exist in the launcher. After a receive,
verify `channel.wait_closed(timeout=...)` before using the message; after a
failed or interrupted startup, check it again after owner termination. If it
returns false, do not reuse the launcher:
terminate or recycle that dedicated process so the outstanding reader and its
memory are discarded. Do not assume closing a pipe from another thread proves
the pending synchronous read was cancelled. Timing is best effort, not a
hard-real-time guarantee. The constructor cleans up tested interruption
boundaries, but Python cannot guarantee atomic cleanup for an asynchronous
exception at every bytecode; recycle the launcher if ownership is uncertain.

An interruption inside process creation can occur after the OS starts a
process but before `Popen` returns its handle. In that uncertain case the
supervisor cannot prove that it terminated or reaped the unknown child. The
marker remains; terminate or recycle the dedicated launcher, and require an
external process and device assessment before treating the device as safe.
The owner contract requires a parent watchdog to start before CUDA access,
but that cooperative watchdog is not independent proof of process teardown.

Marker and attestation files are created exclusively and their contents are
flushed with `fsync`. This supports persistence through ordinary process
termination. Directory-entry durability across OS crashes or sudden power loss
has not been established, particularly on Windows, and is not guaranteed by
this implementation. Following such failures, require an explicit external
device-safety assessment before reuse; there is no automatic marker reset API.

## Read-only inspection of an active Windows child

`inspect_live_child(run_id)` duplicates the registered direct child's Windows
process handle under the spawn-custody lock, checks its actual PID and creation
time, and returns a capability with Windows `SYNCHRONIZE` and
`PROCESS_QUERY_LIMITED_INFORMATION` rights. `check_live()` rejects a retired or
replaced spawn record, a changed PID or birth time, and an exited process.
Keep the capability through identity-sensitive observations, compare its run
ID and PID to the eventual `SupervisionResult`, then close it. This inspection
does not wait for, signal, recover or change the child or its marker. It does
not cover escaped or unrelated processes, authenticate a hostile owner, or
prove physical GPU exclusivity or CUDA cleanup.

Capabilities are rooted before native handle duplication, including when a
Python interruption occurs before the caller receives the result. They are
listed by `pending_live_child_inspections()`; close only capabilities you own
or whose user is known to be quiescent. If native close is interrupted or
fails, its outcome may be uncertain and the capability stays rooted; do not
retry that handle number because it may have been recycled. Recycle the
dedicated launcher if this leaves resource custody uncertain. This mechanism is
distinct from `pending_supervised_children()` and its quiescent recovery API.

The duplication and Windows identity/liveness calls occur while the spawn
registry lock is held. If a native call stalls, supervisor retirement can
also stall; the in-process API supplies no hard deadline. Use a separate
launcher-process watchdog and preserve the marker whenever its outcome is
uncertain. Do not infer successful retirement from a timed-out caller.
