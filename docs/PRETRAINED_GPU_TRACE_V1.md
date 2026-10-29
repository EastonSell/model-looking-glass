# Proposed supervised GPU trace correctness v1

This is a preparation protocol, not a recorded GPU result or an authorization
to run. The exact five-token workload has no measured GPU resource profile yet.
The two-token service bootstrap and eight-token latency benchmark describe
different workloads and cannot authorize this one. The GPU path remains
disabled until the prerequisites below are independently reviewed and met.
`looking_glass.trace_gpu_owner` currently checks a proposed seven-phase key
contract and resolves matching profiles; its public launcher always refuses to
run. It contains no production owner, trusted live-key builder, or measured
GPU profile. A private lifecycle seam is disabled by default; CPU tests
temporarily enable it with fakes to exercise ordering, reservation, and
fail-closed cleanup. Injected collaborators can execute arbitrary code and
assert arbitrary facts, so the seam cannot itself establish device safety or
enforce CPU-only behavior. Its process-local entry guard allows only one owner
run for the process lifetime, including after a successful simulated drain.
Synthetic CPU tests do not establish GPU readiness.

## Frozen work

Use the pinned offline `distilbert/distilgpt2` revision and asset checks from
`PRETRAINED_SMOKE_V1.md`. Run its fixed public token IDs
`[464, 6766, 318, 4171, 13]`, an all-ones int64 attention mask, seed 0,
batch 1 and sequence length 5, in float32, eager/eval/inference mode with
`use_cache=False`. Disable CUDA matmul and cuDNN TF32 as the smoke specifies;
record and match the actual deterministic-algorithm and cuDNN settings rather
than silently changing them. Offload full logits and selected snapshots to
CPU. Run two identical forward
calls each for the uninstrumented baseline, independent reference hooks, and
the `direct`, `gradient`, and `id` adapter modes. Observe attention, MLP and
block outputs at layers 0, 2 and 5 and local tokens 0 and 1. No other inputs,
capture sites, model mutations, precision modes, or concurrent jobs are
included. Required float32 comparisons use the smoke's integer bit-view
equality, including signed zero and subnormal encodings. The reference and
replay are correctness checks, not benchmarks or
causal explanations.

The direct and gradient sessions each retain 18 detached CPU snapshots with
110,592 value bytes, under the existing 131,072-byte per-session cap. The
reference values, direct values, and gradient values can overlap in host
memory; their combined actual value payload is 331,776 bytes, and a proposed
aggregate payload ceiling is 393,216 bytes. These limits do not bound Python
metadata, allocator reservation, temporary copies, gradient workspace, or the
CPU model. A single full float32 logit output is 1,005,140 bytes; six such CPU
arrays can overlap, totaling 6,030,840 logical bytes. A selected 1-by-2-by-768
float32 activation is 6,144 bytes before allocator rounding or gradient
temporaries. Do not present these tensor-size calculations as peak memory.

## Proposed admission and measurement

The pinned model has 81,912,576 distinct float32 parameter elements, or
327,650,304 logical bytes before buffers and allocator overhead. A **proposed,
unmeasured** initial commitment is 512 MiB inclusive for resident model state
and 1 GiB for the combined job and transient workspace, totaling 1.5 GiB. The
existing smoke's 1.5-GiB process allocator ceiling is a separate hard allocator
limit, not proof that this estimate is sufficient; read back and record the
effective fraction after setting it. With an unchanged verified device total,
require the effective byte ceiling `floor(fraction * device_total_bytes)` to
equal the approved 1.5-GiB ceiling. For reserve `max(1 GiB, ceil(15% * total))`
and the sum of all still-active incremental commitments `C`, require both
`device_free_bytes >= reserve + C` and
`device_total_bytes >= process_reserved_bytes + reserve + C`. Sample after
initial synchronization, before weight transfer and during execution under
the actual device lease; reject a changed total or either failed check. This
can conservatively count already allocated memory again while a commitment is
held. A commitment is an estimate, can reject work,
and cannot prevent all out-of-memory failures or control nonparticipating GPU
processes. This proposed amount requires a separate independent review before
it is used; it is not derived from the other workloads' measurements.

The proposed contract uses distinct resource keys for model load; cold and warm
baseline; independent reference capture; direct, gradient, and ID capture.
Bind the actual checkpoint/config assets, exact source and dependency/runtime
identity, physical device, execution mode, input dtype and shape, output cap,
capture policy, ordered preceding phases and host/GPU capture commitments.
Each reference/direct/gradient/ID phase covers its two ordered calls and their
specified earlier retained capture sets. Each future approved profile must
match a live key exactly and be supported by its own preserved measurement
evidence. A separate expected model/runtime identity digest must come from the
trusted live source, asset, runtime, and device verification, not from the
candidate profiles. Checking a caller-supplied digest for consistency does not
authenticate it. The proposed GPU trace allowances account for selected-value staging;
the existing trace cap does not actively enforce a device-wide temporary-memory
bound. Each declared sample must also carry seed 0 and the phase-specific case
digest of the fixed IDs and mask (the load phase has no request inputs). Those
declarations and matching hashes do not authenticate an actual execution;
trusted evidence and live identity must be verified separately. Do not populate a profile with a
borrowed peak or silently extrapolate across lengths or modes.

The guarded seam reserves the entire 131,072-byte GPU and 393,216-byte host
trace allowances up front, within the combined job lease. A future backend must
account its captures against these existing claims rather than claim the same
allowance a second time. Release the claims only after all captured payloads
and dependent output work have been drained.

The proposed per-phase margin is fixed before measurement: 32 MiB plus 10%
of the basis, rounded up to whole bytes. The basis is the declared GPU input,
output and trace allowances plus the maximum across completed samples of zero,
allocated peak minus initial allocation, reserved peak minus initial
reservation, and the decrease in device-free memory between endpoints. The
resulting estimate must fit the 512-MiB load commitment or the 1-GiB job
commitment for every other phase. The margin is an unmeasured heuristic; it
does not establish that unseen workspace or external device pressure is bounded.
The load sample must observe the actual operation, including buffers, temporary
copies, and allocator peaks; parameter-byte arithmetic cannot stand in for that
measurement. Reuse requires a completed profile that fits these commitments;
stop and preserve evidence on any OOM or failed/uncertain attempt.

Before each measured phase, synchronize, sample device-wide free/total and
process allocated/reserved memory, and reset process allocator peaks. After the
phase and output offload, synchronize again and record endpoint device-free
values and allocated/reserved high-water marks. These are distinct quantities:
endpoint free memory is not a device-wide peak, and allocator peaks do not
include all CUDA allocations. Record cold/warm state and the previously retained
host trace sets; repeat or stop according to a separately frozen sampling plan.
Preserve failed, cancelled, out-of-memory, and incomplete attempts as such;
none is a successful profile or a basis for a silent retry.

## Ownership and evidence requirements

Use the same canonical physical-UUID `DeviceIdentity`, shared coordination
root, `run_supervised`, `DeviceLease`, and `OwnerSession` as all participating
GPU work. The directly spawned owner must verify the runtime-reported device
UUID against that identity after obtaining the actual lease and before any
model load. The runner's `lease_id` string alone is not a lock or permission to
access CUDA. A dedicated exact-workload admission, exclusive scheduled window,
and a trusted immutable source/native-runtime snapshot are required. The
source-first child must exclude stale bytecode and unreviewed import/native
library shadowing. Preflight the checkpoint, output location, committed source,
and launch environment on CPU before creating the owner.

Keep the resource commitments, actual device lease, and model ownership until
the workload and offload have completed. Attest clean only after model/capture
references and all output work are drained, synchronization succeeds, live
commitments and allocator counters have been reconciled, the device/runtime
binding is rechecked, and the owner is ready to exit without further GPU work.
The parent must reap that exact child and verify the supervisor's clean result.
An exception, interruption, OOM, failed cleanup, ambiguous binding, missing
attestation, or uncertain owner exit before a valid clean attestation leaves
the sticky device marker in place; do not automatically clear or reuse it. If
an interruption follows a valid post-drain attestation, the supervisor may
clear the marker after reaping the exact child and verifying that attestation,
even when the workload exit status is nonzero. A workload result and verified
teardown are separate facts. Record phase-specific measurements and failure diagnostics,
review any private paths/identifiers before publication, and never overwrite a
prior result.

This protocol remains disabled pending: reviewed admission and worker lifecycle
fixes; the common supervisor and source-first launcher; a trusted live
resource-key builder; an independently reviewed five-token commitment and
measurement procedure; the isolated two-token bootstrap and service smoke; and
a separately scheduled exclusive window for this workload. External GPU users,
physical device provenance, native-library coverage, and asynchronous failure
recovery remain limitations of this cooperative system.
