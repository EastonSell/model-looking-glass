# Cooperative GPU device lease

`looking_glass.device_lease` provides a cross-process exclusive lease for
programs that voluntarily follow the same protocol. It does not inspect the
GPU, restrict VRAM, schedule jobs, or control the Windows desktop and other
external GPU users. A lease does not by itself make benchmark measurements
isolated. Check external GPU activity and report the measurement conditions.

## One lock identity per physical device

Configure every participating process with the **same absolute path to the
same physical file** in a shared, trusted local directory outside individual
worktrees. For example, the current workspace can use
`work/gpu-device-0.lock`. This example denotes a physical device selected by
the operator, not an automatically discovered CUDA ordinal. Processes with
different `CUDA_VISIBLE_DEVICES` mappings must still agree on the same lock
identity for the same physical GPU. Different lock files do not coordinate.

The parent directory must exist before acquisition. Keep the lock file in
place between runs: do not unlink, rename, replace, or recreate it while any
participant might hold or wait for it. The implementation verifies the
opened file's identity on acquisition and on `check()`/`release()`, but cannot
prevent another program from replacing the file after a check. Protect the
containing directory from such changes. All participants must use a local
filesystem whose locking implementation enforces the selected OS primitive;
network filesystems, containers, and cross-OS mounts require separate
validation and are not established by these tests.

Windows uses a nonblocking exclusive byte-range lock at offset zero via
`msvcrt.locking`; POSIX uses nonblocking exclusive `flock`. Do not mix these
different locking mechanisms against the same storage and assume they
coordinate. The tests run multiple independent processes on Windows; POSIX
behavior should be tested on the target system. The OS lock, not file
contents or a PID stored in the file, determines ownership. The file is never
removed or populated with stale owner metadata by this implementation.

## Lifetime and failure handling

```python
from pathlib import Path
from looking_glass.device_lease import DeviceLease, LeaseError

# Supplied by the supervisor, identical for broker and benchmark processes.
lock_path = Path(r"C:\shared-workspace\gpu-device-0.lock")
try:
    lease = DeviceLease.acquire(lock_path, timeout=30.0, cancel_event=None)
except LeaseError:
    # Do not initialize CUDA or accept GPU work.
    raise

# Only now create the CUDA-owning worker/model and run device work.
# A dedicated owner process should keep `lease` for its entire lifetime and
# terminate after stopping admission, draining work, and synchronizing.
```

`timeout` is required, finite, and nonnegative; zero makes one immediate
attempt. An optional cancellation event must provide `is_set()` and `wait()`.
`LeaseTimeoutError` and `LeaseCancelledError` distinguish busy and cancelled
acquisitions; both inherit from `LeaseError`. Unexpected lock, identity, or
platform errors also fail without returning a lease. Waiting for the internal
mutex and retrying an OS lock honor the deadline and cancellation; a stalled
filesystem syscall or operating-system scheduling can exceed wall-clock
deadlines, so use a reliable local filesystem and an outer process watchdog
when a strict deadline is required. The lock provides no FIFO fairness.

The broker process must acquire **before** initializing CUDA or loading the
model and retain its lease while any CUDA resources or work remain. For an
isolated benchmark window, stop broker admission, drain and synchronize its
work, close the worker, and wait for the entire GPU-owning process to exit.
The benchmark then acquires the same lease **before** initializing CUDA and
holds it until its own process exits. Shutdown or synchronization failure is
not evidence of a clean drain; retain the lease and quarantine/terminate the
owner process rather than opening an isolated window. The broker and benchmark
must both be wired to this protocol before it enforces exclusion.

The descriptor is non-inheritable across an exec/spawn. On POSIX, the fork
handler closes inherited descriptor copies in the child. A child cannot use
or release its parent's `DeviceLease` object. Dropping a Python reference does
not release the lock; the process holds it until explicit release or exit.
The OS releases locks after process termination, potentially after a delay,
so a successor must still use bounded acquisition and check the environment.

`lease.release()` is available when the caller can establish that all device
ownership and work have already ended. There is intentionally no automatic
context-manager release on an exception: exception handlers and tracebacks may
retain live CUDA objects. If file identity changes, release fails, or another
lease operation reports ambiguous ownership, stop GPU work and terminate the
owner process; do not retry release or treat `check()` as proof of ownership.
Before attempting to unlock, `release()` invalidates the object's ownership
claim. If it is interrupted before cleanup completes, that lease cannot pass
`check()` or be retried, even if the OS has already admitted a successor. The
interruption itself propagates; the descriptor and bookkeeping can remain in
an uncertain state, so terminate the owner process without further device work
or forking. In particular, interruption after closing the descriptor can leave
its integer in the registry, and the OS can reuse that integer for another
descriptor; the fork cleanup must not be treated as safe recovery in that
state. This does not prevent a separate process from acquiring a lock that the
OS has already released.
`check()` only checks process ownership and the named file's identity; it is
not a hardware status check and cannot stop external processes or file
replacement between checks.

References: [Microsoft `_locking` documentation](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking?view=msvc-170),
[Microsoft `LockFileEx` documentation](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex),
and [Python file locking interfaces](https://docs.python.org/3/library/msvcrt.html#msvcrt.locking).
