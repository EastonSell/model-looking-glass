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

Marker and attestation files are created exclusively and their contents are
flushed with `fsync`. This supports persistence through ordinary process
termination. Directory-entry durability across OS crashes or sudden power loss
has not been established, particularly on Windows, and is not guaranteed by
this implementation. Following such failures, require an explicit external
device-safety assessment before reuse; there is no automatic marker reset API.
