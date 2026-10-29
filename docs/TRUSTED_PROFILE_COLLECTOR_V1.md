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
and a private evidence directory outside the checkout. Its explicit HF home
must contain the offline checkpoint cache, and the fixed launcher supplies that
same home to the owner. It uses the shared
fixed launcher and identity-aware supervisor. Collection and the subsequent
service must execute from the same frozen checkout path and environment;
source, dependency, runtime or configuration changes can invalidate the keys.
Both parent and child require the reviewed gated, rooted child-launch custody
contract before attempting ownership. The marker identifies the expected
interface; it does not authenticate arbitrary process code. If native spawn
ends without a usable child handle, the outcome remains quarantined.
Before acquiring ownership, the child rejects a preloaded Transformers or
Hugging Face Hub module, including submodules: the hub caches its home when it
is imported. The fresh child receives its fixed environment before imports.
The child reads the parent-pinned configuration bytes before CUDA, holds the
canonical device lease and verifies the runtime GPU identity. The private
artifacts contain the GPU UUID and absolute paths and should not be published
without redaction.

The collector holds a 2 GiB resident commitment, then admits one 512 MiB
forward commitment before the cold forward. Both remain active and rooted for
all four serialized forwards, cleanup, and clean attestation until process
exit; the forward claim is not retired between observations. It verifies the CPU checkpoint
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

The dedicated collector runs its GPU operations and sample decisions serially
on its owner thread; its IPC reader is joined before a sample can complete.
Owner and sticky-state checks precede output and sample writes and clean
attestation. These are observations, not an atomic guarantee against an
unrelated thread changing an unsafe flag after a check. Other code must not
perform CUDA work or change the owner state concurrently in this process.

After worker shutdown and local disposal, a version-bound PyTorch cuBLAS
workspace cleanup records the before/after counters. The child requires zero
current allocated/reserved bytes, publishes a private candidate and attests
clean while retaining the device lease and both memory commitments until
process exit. The cleanup record includes the active commitment sizes; the
owner rechecks them immediately before attestation. The parent reaps
the exact child, verifies artifact hashes, case/seed order, admission policy,
profile identities and the zero-counter cleanup, and only then publishes a
success terminal. The three profile IDs still require independent review and
trusted configuration before service use; the collector does not approve them.
Failure and partial records do not become measured profiles.
The dedicated child exits with `os._exit` after its durable writes on both
success and failure, so Python `atexit` handlers cannot enqueue GPU work after
attestation or during quarantine. The supervisor observes its process exit.

When the driver inventory subprocess fails, the private failure record may also
contain its command, exit status, up to 4 KiB each of captured stdout and stderr,
and a bounded allowlist of its launch environment. These fields can contain
device identifiers or paths and remain private. The retained-size limit is not
a bound on the subprocess's temporary output allocation. Missing diagnostics
do not establish the cause, and collection does not retry or clear quarantine.

CPU tests use fake device callbacks and a real CPU worker/IPC frame; they do
not establish GPU behavior. The versioned protocol also records source and
runtime identity, but does not authenticate all native dependencies or prove
the system is a security sandbox.
