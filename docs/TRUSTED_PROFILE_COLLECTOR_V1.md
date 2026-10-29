# Trusted-service resource collector, v1

This collector is intended to measure the fixed worker path described in
[the trusted-service protocol](TRUSTED_SERVICE_PROFILE_V1.md). It is separate
from the direct-model GPU one-shot and is not a latency or throughput test.
It records one model load (seed 17), one cold worker forward (seed 17), and
three warm forwards (seeds 29, 43 and 71) for the exact two-token input.
No capture or trace memory is admitted. A successful result is a measured
estimate with predeclared margins, not a guarantee against future OOM.

`run_supervised_collector` requires a clean detached checkout, a verified
offline checkpoint, a common physical-device workspace, a trusted environment
and a private evidence directory outside the checkout. It uses the shared
fixed launcher and identity-aware supervisor. Collection and the subsequent
service must execute from the same frozen checkout path and environment;
source, dependency, runtime or configuration changes can invalidate the keys.
Both parent and child require the reviewed gated, rooted child-launch custody
contract before attempting ownership. The marker identifies the expected
interface; it does not authenticate arbitrary process code. If native spawn
ends without a usable child handle, the outcome remains quarantined.
The child reads the parent-pinned configuration bytes before CUDA, holds the
canonical device lease and verifies the runtime GPU identity. The private
artifacts contain the GPU UUID and absolute paths and should not be published
without redaction.

The collector holds a 2 GiB resident commitment and separately admits a
512 MiB commitment for each sequential forward. It verifies the CPU checkpoint
and conversion policy, copies the model, synchronizes, checks tied device
parameters, synchronizes again, and executes `ResidentWorker.execute` itself.
The callback constructs the fixed input privately, calculates its actual case
digest, verifies the finite bounded CPU output, and measures a representative
frame through the production IPC encoder/decoder. The frame check establishes
its envelope, layout, finite value count and size for that fixed envelope; it
does not establish numerical parity or a bound on every possible response.

The saved samples contain synchronized process allocator snapshots and peaks,
and separate device-wide free/total snapshots. Every completed stage and its
admission are written before proceeding. A callback failure, OOM, timeout,
unverified drain or lost owner retains custody and prevents clean attestation;
the supervisor's marker remains for assessment. The collector will not retry a
CUDA synchronization after the worker or owner is marked unsafe. An abrupt
interrupt between a model conversion call returning and assignment can still
drop an unrootable temporary: the owner must terminate promptly, and the
marker cannot be treated as clean in that case.

After worker shutdown and local disposal, a version-bound PyTorch cuBLAS
workspace cleanup records the before/after counters. The child requires zero
current allocated/reserved bytes, publishes a private candidate and attests
clean while retaining the device lease until process exit. The parent reaps
the exact child, verifies artifact hashes, case/seed order, admission policy,
profile identities and the zero-counter cleanup, and only then publishes a
success terminal. The three profile IDs still require independent review and
trusted configuration before service use; the collector does not approve them.
Failure and partial records do not become measured profiles.
The dedicated child exits with `os._exit` after its durable writes on both
success and failure, so Python `atexit` handlers cannot enqueue GPU work after
attestation or during quarantine. The supervisor observes its process exit.

CPU tests use fake device callbacks and a real CPU worker/IPC frame; they do
not establish GPU behavior. The versioned protocol also records source and
runtime identity, but does not authenticate all native dependencies or prove
the system is a security sandbox.
