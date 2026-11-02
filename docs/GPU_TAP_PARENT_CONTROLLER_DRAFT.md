# Repeated tap parent controller, draft

This CPU-only checkpoint is **disabled and not an executable GPU launcher**.
It contains a private driver for preconfigured supervision, live-child
identity, activity polling and the two barriers. The source-bound public
entrypoint and actual CUDA owner are not wired; the integration and evidence
replay still require review.

The intended ready frame is a single restricted supervisor startup-channel
message. It names the fixed session slot, the owner's PID and Windows creation
time, the supervisor run ID, a fresh parent challenge, a monotonic timestamp
and clock metadata. It also carries the owner's canonical physical UUID and
exact CUDA total, which the parent binds to the supervised device identity.
The parent validates exact fields and duplicates the
supervisor's registered child handle, checking its PID, creation time, live
registration and the canonical marker. Both sides must use the pinned Windows
CPython 3.11.15 runtime and report `QueryPerformanceCounter()` for
`perf_counter_ns`; Python documents this clock as system-wide on Windows since
3.10 ([Python 3.11 time documentation](https://docs.python.org/3.11/library/time.html#time.perf_counter)).
The reciprocal frame/grant timestamps must respect that time base with a
conservative two-reported-tick separation for cross-process ordering; Microsoft
notes that timestamps from different threads within one tick have ambiguous
order ([Windows QPC guidance](https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps)).
These runtime and ordering checks do not prove physical execution or resist a
hostile local process. The later child grant check and saved-journal replay must
also bind the actual first preflight timestamp and the same clock time base.

The parent takes three GPU points and a process query after startup and
requires them to be valid and fresh. Before issuing a grant it compares the
owner's exact CUDA total with the stable NVIDIA-SMI total, allowing at most
the protocol's 2 MiB cross-provider difference. The private grant binds both
totals and the UUID; the owner must recheck these before preflight. Agreement
is a consistency check, not proof of physical identity or an admission bound.
The start grant carries the oldest GPU
and process query start times and expires no later than 500 ms after issue or
two seconds after either older query. The owner must independently enforce
those boundaries at its actual preflight start; a stale grant must not permit
the first forward. Its one-use preflight gate authorizes an interval of at most
250 ms, with the grant expiry and two-second telemetry freshness checked at
the upper end. The actual invocation timestamp and both interval endpoints
are saved and rechecked against the grant before cleanup and after reap.
Device synchronization and memory-snapshot preparation precede the ready
frame; the owner rechecks its session and admission before invoking the model.
On Windows the file grant is written, flushed and published
with a same-directory rename that refuses an existing destination; the visible
file has one link. The draft POSIX fallback is not an enabled GPU execution
path. These files and the environment challenge are private cooperative
coordination, not authentication against another same-user process. A grant
already made visible may be consumed if the parent is interrupted just after
publication; future integration must retain supervision and treat any
ambiguous attempt as incomplete rather than assuming no forward occurred.
The child barrier reader retries only missing files and bounded Windows sharing
violations until its deadline; malformed or stale content fails closed. Its
preflight authorization is one-use and performs no I/O or device work.

The intended finish barrier uses the actual source-stamped final memory
postcheck completion, not the last timed wall end; it must match the journal
and occur within two seconds of that wall end. The owner must perform no more
CUDA-facing work until the finish grant. After the finish stamp, the process
query and three eligible idle points can extend to five seconds after the
stamp, or seven seconds after the last measured wall end.

The proposed owner closes its 86-line canonical private JSONL journal before
publishing the finish frame. The bounded parent reader hashes its bytes and
matches the initial preflight and final postcheck to that frame before a
cleanup grant. This check does not validate the other journal rows. After the
child is reaped, the driver rehashes the same bytes and runs the v1.1 schedule,
memory and paired-statistics replay. The
closed-file and source-stability assumptions are cooperative, not filesystem
leases or proof of execution.

The parent writes and fsyncs its activity journal while the owner may be
inside a measured wall span. That disk and serialization work is part of the
observed host context and may affect timing. The collector retains sanitized
failure codes and counts, but not necessarily failed query timestamps, raw
stdout or stderr. If a producer cannot be joined, the controller retains its
attempt and supervisor custody and cannot report a complete screen.

Only a successfully completed full implementation may bind this draft to an
owner. The public timing result must distinguish project coordination from
physical exclusivity and preserve any failed or incomplete attempt. No
forward, checkpoint load, CUDA access or physical-device operation is performed
by this checkpoint or its tests.
