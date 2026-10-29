# One-shot five-token pretrained CUDA trace pilot

This prospective, single-attempt measurement implements the fixed workload in
`PRETRAINED_GPU_TRACE_V1.md` and the bitwise comparisons in
`PRETRAINED_SMOKE_V1.md`. It measures the previously unprofiled five-token path;
it does not enable the disabled shared trace owner, approve a reusable resource
profile, provide a timing benchmark, or establish model causality. Publish the
exact committed source and this amendment before its first GPU forward.

Use the pinned offline checkpoint and input `[464,6766,318,4171,13]`, an
all-ones mask, float32 eager evaluation, inference mode, seed 0, no cache and
one CPU thread. Disable CUDA matmul and cuDNN TF32. Record actual deterministic
algorithm and cuDNN settings; check them at the scientific boundaries. The ten
forwards are ordered as one cold baseline, one warm baseline, two independent
reference-hook forwards, and two each for direct, gradient and ID tracing.
Here "cold" means the first model forward after checkpoint loading, CUDA setup
and input transfer; it does not mean a cold process or device, and neither
baseline is a latency measurement.
Capture layers 0, 2 and 5 at attention, MLP and block outputs, positions 0 and
1. The independent hooks and the trace adapter each record 18 ordered events
over their two calls. Each value-bearing set is exactly 110,592 float32 bytes;
ID events contain no values. Match the full logits and all corresponding
reference/direct/gradient snapshots by integer bit view, including signed zero
and subnormals; reject any mismatch, nonfinite result, bad metadata or cache.
These are within-run checks, not a CPU/GPU equality claim.
The gradient readout differentiates an auxiliary dot product with respect to a
fresh probe and returns the selected activation. Its parity with direct capture
does not establish a gradient of the model output or loss.

For this unmeasured one-shot, independently review and admit the existing
proposed 512 MiB resident commitment plus 1 GiB job commitment under the
canonical physical device lease. Set and read back an effective 1.5 GiB
PyTorch allocator ceiling. Reserve 131,072 GPU trace bytes and 393,216 host
trace bytes for the entire run; these are logical custody claims, not bounds
on native allocations, allocator reservation, workspaces or process RSS.
Require the original reserve and both free/total inequalities from the fixed
protocol at admission and at each phase boundary. Keep the commitments and
known model, input, output, session and capture roots through synchronized
completion; failed or uncertain work cannot authorize another phase.
After the load phase, the fixed 80 input bytes are transferred and
synchronized separately, before the cold baseline; that setup is recorded and
the input bytes are included in each subsequent phase's allowance.

For load, cold baseline, warm baseline, reference, direct, gradient and ID,
synchronize and record device free/total plus process allocated/reserved
before the phase, reset allocator peaks, then synchronize after work and CPU
offload and record endpoint free/total and allocated/reserved peaks. Record
the fixed case, phase order, retained value sets and a conservative estimate
using the original 32 MiB plus 10% margin rule. A measured phase that exceeds
its proposed commitment is incomplete. Endpoint free memory is a sampled
device-wide point, not a peak; allocator peaks exclude other processes and
native CUDA allocations. No observation here is a capacity guarantee.

The checkpoint is read from the existing verified offline cache, with its
verified staging copy, attempt files and temporary caches on E:. Check the
output-volume and workspace-volume free space separately before launch. Keep
the canonical lease and marker in the shared C: workspace. Checks and
commitments are cooperative and cannot stop external GPU or disk activity.
The diagnostic JSON records have per-file limits. The retained controller
stdout and stderr files are uncapped trusted-process output; the tensor claims
are not disk, RSS or native-allocation quotas.
Source-first isolation, a retained direct-child handle, runtime UUID matching,
guarded synchronization, model/session disposal and a clean supervisor
attestation are required. An exception, OOM, cancellation, failed check or
uncertain exit preserves bounded partial records and known live roots, leaves
the unsafe marker when cleanup is unverified, and is not retried automatically.
Any separately authorized successor attempt requires a new output directory,
qualified resolution of the prior marker and owner, reviewed fixed source and
a fresh scheduled admission. Preserve the earlier incomplete record; it is
not a measurement or evidence of successful cleanup.
The observer and controller use the direct base interpreter. Before controller
work, the child verifies the parent and child IDs carried by a private launch
pipe and writes an identity receipt; the observer records whether that receipt
matches its retained handle. A lost or partial pre-grant spawn cannot start the
controller workload. This does not prove the absence of untracked descendants.
Source verification compares checkout bytes with the recorded Git commit at
startup and again after work. It assumes a trusted, quiescent checkout and
native libraries; it cannot rule out a change and restoration between checks.
Owner checks are point checks: a loss during one in-flight model/hook operation
can be detected only at its next boundary. No GPU cleanup or later workload
is attempted after a detected unsafe boundary.

Publish the saved phase/correctness records and independent source, arithmetic
and cleanup review only after the exact owner has been reaped. Public records
omit physical UUIDs, local paths and process identifiers. The fixed smoke's
hash-only evidence policy remains in force: do not publish raw logits,
activations or model weights. A successful record supports only this one input,
device and execution, not a reusable resource approval or a claim about other
models or contexts.
