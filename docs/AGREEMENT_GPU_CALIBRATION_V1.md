# Proposed GPU calibration for five-token agreement interventions

This is a **draft protocol and CPU-checked payload calculation**. It is not a
GPU result, an approved resource profile, or authorization to run the current
CPU-only agreement executor on CUDA. The separately reviewed GPU adapter and
its policy digest must be published before this protocol can be frozen. The
two-token no-capture service profile and the direct-model one-shot are not
compatible with this workload.

The intended workload is the pinned DistilGPT2 model, one five-token int64
input and an all-ones int64 mask, float32, eager evaluation, inference mode,
no cache, and the actual ordered intervention path. Candidates are projected
attention and MLP outputs at layers 1–4, plus last-token and subject-token
controls at the final block. Each forward materializes the full
`[1, 5, 50257]` logits before the adapter's bounded host offload. The source,
model, runtime, physical device, capture policy, exact callbacks and limits
must be bound into workload keys independently of the two-token service.

Before the first CUDA operation, freeze a private resource-only run manifest
with the exact source commit and checkout, checkpoint hashes, runtime/driver,
device UUID, seed `20261028`, and the previously exposed public five-token
target `[464, 3797, 1474, 262, 9168]` and donor
`[464, 11875, 1474, 262, 9168]`. Subject index is 1 and last-token index is 4.
Before the 13 samples, execute and separately record a same-owner warmup using
the target row, seed `20261028`, and the same ten-site capture and offload
path. Bind those details to the adapter's reviewed warmup-protocol digest; a
profile labeled `warm` alone does not establish that warmup occurred.
Capture the ten donor vectors anew under the measured GPU runtime and retain
their host storage under a live trace claim until all actions finish. This
calibration does not generate or inspect held-out cases, and it retains no
behavioral logits or effect estimates. The resource-only sequence requires one
completed sample of each kind, in this exact order:

1. `clean_capture`: the ten donor-site captures and offload.
2. `layer1_attn_cross_patch`, then `layer1_mlp_cross_patch`.
3. `layer2_attn_cross_patch`, then `layer2_mlp_cross_patch`.
4. `layer3_attn_cross_patch`, then `layer3_mlp_cross_patch`.
5. `layer4_attn_cross_patch`, then `layer4_mlp_cross_patch`.
6. `null`, then `self_patch`, both at layer-1 attention, last token.
7. `block_last_cross_patch`, then `block_subject_cross_patch`, at layer 5.

Each action uses the fixed target and full selected/whole pre/post witnesses;
cross-patches use the corresponding freshly captured donor site and coordinate.
The attention/MLP candidates use the last token; the two block controls use the
named coordinate. A case digest must bind the exact target and donor IDs,
module/layer, tuple selector, invocation, batch/token coordinates, operation,
donor source and limits. Agreement with a label alone is not sufficient.

Model-load and model-state hashing/offload measurements are separate and need
their own resource and host-custody accounting; the 13 action samples do not
bound those paths. The manifest must identify the
specific modules and token coordinates; a generic list of sample labels does
not establish that the intended actions ran. One sample per kind is an initial
measurement, not evidence of a worst-case bound across layers, inputs, CUDA
kernels, drivers, or future execution. If the actual executor or policy
differs, use a new protocol and remeasure before admitting that work.

## Payload accounting

These are exact *logical tensor payloads* for the fixed shapes and dtypes:

| Payload | Bytes |
| --- | ---: |
| Input IDs and mask together | 80 |
| One 768-element selected vector | 3,072 |
| One complete `[1, 5, 768]` activation | 15,360 |
| Full five-token vocabulary logits | 1,005,140 |
| Ten selected clean captures | 30,720 |
| Action captures: whole before/after plus selected before/after | 36,864 |
| Full activation clone made by a patch or null action | 15,360 |
| One cross-patch source clone inside the plan | 3,072 |

The current CPU executor uses at most ten steps for clean capture and five for
an action. Its plan limits are 1 MiB per tensor, 512 KiB aggregate captures,
128 KiB aggregate sources, 16 steps and one invocation per module. The
proposed adapter reserves **2 MiB each** for GPU and host trace storage and
must enforce those allowances for all retained objects and transfers. The
logical sizes above are not a proof that 2 MiB suffices: the caller's donor,
plan clones, pre/post captures, offload copies, temporary tensors, allocator
rounding and storage aliases may coexist. Model weights, intermediates, full
logits, CUDA workspaces, native allocations, host serialization and external
GPU usage also lie outside a simple sum of these payloads. The CPU helper
`agreement_calibration_spec` checks only necessary plan minima and proposed
profile declarations; it does not enforce device or host storage at runtime.

## Measurement and admission

Use a supervised short-lived owner, the common physical-device lock and a
verified runtime CUDA UUID. Record a fresh free/total snapshot under the lease,
and use the native PyTorch allocator with a 25% per-process allocator fraction;
this is a ceiling on that allocator, not on every CUDA or driver allocation.
Keep the fixed load and job commitments held through the corresponding work,
use the shared budget's free/reserved checks and a reserve of at least
`max(1 GiB, 15% of total)`, and reject if the live device cannot admit them.
The exact load/job commitments must be frozen with the reviewed adapter before
measurement; an old estimate is not a substitute for them.

Synchronize before each memory baseline, reset peak counters, execute the
actual adapter action including capture, patch and host offload, then
synchronize before recording allocated/reserved peaks and device-wide
free/total. Save the raw readings, caps, sample kind, seed, case and source
identity even on failure where safely possible. Build a profile only from
completed samples covering all 13 kinds, using at least a 512 MiB fixed
margin plus 20% of the declared-plus-observed basis. Independently recompute
the arithmetic and verify the case, source, sample order and cleanup before
the profile IDs are configured as approved. The margin is a policy choice, not
a guarantee against OOM or device-wide interference.

On cancellation, OOM, lost ownership, synchronization failure or uncertain
disposal, stop new work, keep the known model/trace/lease references until the
owner exits, and preserve the unsafe marker. A clean result requires verified
drain, disposal, the version-bound cleanup policy and a matching reaped child;
do not infer CUDA-context or external-process cleanup from PyTorch allocator
counters alone. No such GPU run may start until the shared supervisor's
spawn-interruption fix is independently reviewed and merged and the GPU window
is coordinated.
