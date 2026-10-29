# Planned supervised grouped-GPU development check

This is a plan, not a completed experiment or a frozen executable protocol.
The grouped broker currently has CPU integration evidence only. A reviewed
runner, independently retained protocol and source hashes, and a dedicated
coordinated GPU window are required before collecting these observations.
GPU launch is explicitly disabled in this version: the session-bound worker
transition and device-lease interruption fixes must be integrated and reviewed
before enabling it. The synthetic CPU tests do not authorize a GPU run.

The proposed first check uses one resident
`distilbert/distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa`, float32,
evaluation/inference mode, eager attention, disabled KV cache and autocast,
with a fixed seed of `2026102808`. Each request supplies one four-token
`int64` row and an all-ones `int64` attention mask. Only two requests from
different teams may form one group, for a combined shape of `[2, 4]`; no
traces or interventions are enabled. The model returns float32 logits of
shape `[2, 4, 50257]` (1,608,224 logical bytes). The resident output limit is
2 MiB and its combined input limit is 128 bytes.

The previously public rows `[50256, 103, 203, 303]` through
`[50256, 106, 206, 306]` provide two mixed-owner GPU sample groups for a
fresh GPU-specific serial-versus-grouped policy. Those rows were already used
for CPU development and are not independent held-out evidence. Two additional
groups use `[50256, 107, 207, 307]` through `[50256, 110, 210, 310]`, held
out from policy construction, to exercise broker routing against separately
run serial GPU references. These are small synthetic samples, not a claim of
batch separability outside the tested inputs. Comparison is elementwise
`abs(grouped - serial) <= 1e-6 + 1e-5 * abs(serial)`; nonfinite output,
missing rows, or any violation fails the check. Failure records must be
preserved rather than replaced by a retry with different tolerances.
The generic validator names its exclusion field `discovery_row_hashes`; here
it contains the later broker rows solely to enforce disjointness. The policy
groups themselves are previously exposed samples, not independent evidence.
Each broker cohort must have distinguishable serial references under the
frozen tolerance, and each delivered row must match its own reference and not
the other. Record team-scoped request and distinct broker job identities,
input-row digests, and exactly one observed model entry per call.

The proposed one-process run must acquire the existing canonical physical-GPU
lease through `OwnerSession`, verify the actual CUDA runtime UUID against the
leased device before importing Transformers, and use a 25% native PyTorch
allocator fraction. Require at least 5 GiB device-wide free before model load.
A provisional 2 GiB resident
commitment and 1 GiB combined-forward commitment limit admission for this
short experiment; these are conservative declarations, not measured
worst-case guarantees or authorization for other shapes. Synchronized
device-free/total and process allocated/reserved/peak counters must be saved
before loading, after loading, and at each serial and grouped forward. Hold
the lease until verified synchronization, disposal and supervised teardown.
Other applications may allocate concurrently, so out-of-memory must remain an
explicit failure and trigger the established quarantine path.
The minimum reserve is 0.5 GiB, while actual admission uses the greater of
1 GiB, 15% of device memory, and that minimum. Record the actual reserve,
existing commitments and remaining admissible estimate at each admission;
the preload threshold does not guarantee later admission.

The broker's exclusive host ledger must satisfy its constructor headroom
formula. An initial 64 MiB / 256 item / 8 request / 16 operation ledger is
proposed for at most four pending jobs and two retained two-request cohorts;
the exact worker envelope and validation scratch requirements must be checked
before acceptance. Serial-reference outputs and comparison temporaries use a
separate 32 MiB control ledger, claimed before input tensors are constructed.
Python objects and any persisted or serialized buffers need separate accounting;
these allocations must not be described as covered by the broker's ledger.
Use bounded chunkwise comparisons or a separately profiled comparison
envelope, and record the chosen bound before starting.

Before execution, freeze the exact inputs, source revision and source hashes,
checkpoint hashes, complete measured runtime/numerical manifest, shapes,
seed, limits, tolerances and expected artifact schema. Build the fresh policy
from GPU observations under that exact runtime, then submit the held-out rows
through the real broker and verify the observed model-entry shape, member
ordering, cancellation/failure status and output association. A CPU policy
must never be reused as GPU certification. The run is a functional and memory
development check, not a latency or throughput benchmark; any throughput
comparison needs the separately specified paired measurement protocol and
its own coordinated timing window.

The controller and child use `scripts/grouped_gpu_dev_bootstrap.py` with
`-I -S -B`, an empty absolute `-X pycache_prefix`, and an explicitly supplied
`--site-packages` directory outside the checkout. Project modules load only
from an allowlisted set of source files; sourceless dependencies are rejected.
The runtime contract pins PyTorch 2.8.0+cu128 and Transformers 4.57.3 and
checks deterministic, precision, allocator, autocast, cache and thread settings.

The supervised owner holds the physical device lease and the conservative
resident commitment through process exit.
Each driver-level CUDA operation checks the owner before and after the call;
failure or an unreadable worker health state is rooted and prohibits later
driver CUDA calls. An operation already inside the worker can complete and
release its forward accounting before the postcheck. This dedicated-owner
protocol does not establish an atomic shared-service retirement contract.
Known exception and returned model references are rooted on failure; repeated
asynchronous interruption, a return interrupted before assignment, inaccessible
runtime temporaries and arbitrary caller aliases require process recovery.
The complete runtime manifest and selected physical identity stay in the
private run directory; the public record contains a redacted manifest and
hash/size inventory. Pass promotion requires a clean reaped owner and
reconciles saved source/protocol, checkpoint, sampled policy, owner identity,
per-call entries, routing, memory and cleanup records. It checks consistency,
not independent numerical recomputation: raw logits are not retained.
The terminal result uses create-only publication. An interruption can leave a
complete file even though the controller did not return; release or audit must
also verify the controller's successful exit or caller acknowledgement. Keep
an ambiguous file as evidence and never overwrite or retry its terminal path.
