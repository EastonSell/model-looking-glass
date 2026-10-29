# Planned supervised grouped-GPU development check

This is a plan, not a completed experiment or a frozen executable protocol.
The grouped broker currently has CPU integration evidence only. A reviewed
runner, independently retained protocol and source hashes, and a dedicated
coordinated GPU window are required before collecting these observations.

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

The proposed one-process run must acquire the existing canonical physical-GPU
lease through `OwnerSession`, verify the actual CUDA runtime UUID against the
leased device, and use a 25% native PyTorch allocator fraction. Require at
least 3 GiB device-wide free before model load. A provisional 2 GiB resident
commitment and 1 GiB combined-forward commitment limit admission for this
short experiment; these are conservative declarations, not measured
worst-case guarantees or authorization for other shapes. Synchronized
device-free/total and process allocated/reserved/peak counters must be saved
before loading, after loading, and at each serial and grouped forward. Hold
the lease until verified synchronization, disposal and supervised teardown.
Other applications may allocate concurrently, so out-of-memory must remain an
explicit failure and trigger the established quarantine path.

The broker's exclusive host ledger must satisfy its constructor headroom
formula. An initial 64 MiB / 256 item / 8 request / 16 operation ledger is
proposed for at most four pending jobs and two retained two-request cohorts;
the exact worker envelope and validation scratch requirements must be checked
before acceptance. Serial-reference outputs, comparison temporaries, Python
objects and any persisted or serialized buffers need separate accounting;
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
