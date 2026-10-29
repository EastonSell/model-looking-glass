# Parent activity points, draft

`tap_repeated_activity` is an independently usable parent-side parser and
bounded point inventory for the proposed repeated tap experiment. Its CPU
tests use fake query output; they do not access CUDA or query a physical GPU.
It is not yet wired to the future launcher, start gate, owner or replay.

The private `_NvidiaSmi` provider runs fixed `nvidia-smi` GPU and compute-process
queries without a shell, with a short timeout and a hidden Windows console.
The source executable and runtime environment are trusted and must be
identified separately by the future parent. Output is checked against a
64-KiB retained limit **after** subprocess capture; process pipe buffering,
command output before that check, and driver allocations are not physically
capped by the collector. A timeout or interrupted subprocess may require an
external process/thread watchdog. The provider must run in the parent, not
the timed owner's CUDA process tree.

Each point is bounded by parent `perf_counter_ns` query start and end times.
GPU memory counters from the command are reported in whole MiB and converted
to bytes. The prospective v1.1 policy retains the CUDA and NVIDIA-SMI totals
separately, checks stable per-provider readings, and uses a conservative 2 MiB
maximum cross-provider difference; this does not prove physical identity. The process query can
miss WDDM/graphics or transient activity. PID, process name and physical UUID
remain in private records only. Parser failures, query errors, duplicate
process phases, overlap, overflow, an in-flight seal, or late calls make the
collector incomplete. Control-flow interruptions propagate after the local
failure marker when possible.

The inventory retains at most 512 points by default, up to 32 error details
plus an error count, and at most one successful process query per phase. It
does not create its own polling thread. The future parent must start and stop
the producer, join it, and retain/query the collector's final state before
sealing; a caller's `producers_stopped=True` assertion is not independently
verified. An old sealed copy cannot be retroactively invalidated by a later
caller. A stuck producer cannot be treated as a complete screen. Actual owner
startup must wait for a live-handle identity, fresh initial points and process
query, and a separately reviewed parent grant before any preflight forward.
`snapshot()` returns detached bounded records while collecting; it reports
in-flight work as incomplete and cannot guarantee later producer success.
The pending marker stays set until a query result or failure is recorded, but
the parent must still detect escaping producer errors and join all producers;
an interruption at entry or a stranded lock requires external recovery.

The prospective v1.1 screen in `tap_repeated_protocol` treats any eligible
idle point above 5% as contamination, and separately reports unavailability.
The public projection contains every valid point's time, utilization and
memory values, without UUID, PID or process name. This collector does not
retain failed or malformed query contents, stderr, or necessarily their
timestamps; it retains bounded sanitized error codes/counts and makes the screen
unavailable. No sample sequence proves continuous
absence of other work or a physical GPU reservation.
