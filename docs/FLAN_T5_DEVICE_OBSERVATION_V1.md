# FLAN-T5 device observation boundary

This is a building block for a future source-bound FLAN-T5 CUDA experiment,
not a launcher, a completed resource profile, or permission to use the GPU.
The existing CPU studies and their evidence remain unchanged. The CPU entry
only exercises the boundary on synthetic T5 models; no CUDA run is implied by
its tests.

The observation runs one eager, evaluation-mode, uncached T5 root forward at
batch size one. Source and target IDs have a single final EOS, no padding,
and bounded lengths of 32 and 8; the decoder IDs are `[0] + target[:-1]` and
both masks are explicit. Supported actions are an unhooked clean forward, a
full-state capture, and ordered capture/action/capture at
`encoder.final_layer_norm`, invocation zero. Patch, null and zero can act on
one token or the complete state. The result retains the complete logits,
encoder state and, for hooked actions, complete before/after witnesses on CPU.
The stored source/decoder IDs and masks describe the actual transferred
inputs. Integer views are copied before interpreting float32 values, preserving
their bit patterns, including signed zeros and subnormals. Source tensors and
all returned arrays must be treated as quiescent by the caller.

The observer validates exact plan coordinates and metadata, finite outputs,
input preservation, patch landing, null and zero controls, and unchanged
unselected positions. Returned device tensors must be contiguous before the
bounded copy; noncontiguous CPU patch sources are accepted. It compares
captured and root encoder states bitwise.
These within-run checks do not require or imply equality between CPU and GPU
outputs. Scores can be calculated on the returned CPU logits under a separately
frozen study protocol; the observer does not choose targets, compare sites, or
make scientific conclusions.

The CUDA entry requires an already validated `OwnerSession`, its exact held
physical-device lease, a runtime UUID binding, and a dedicated active
`MemoryLease` with at least 512 KiB of device trace claims and 2 MiB of host
trace claims. Each memory-lease object is consumed for at most one observation
attempt, including under concurrent calls. The observation claim must remain
held until all returned host storage, its aliases and any device objects have
actually been disposed; on uncertainty, retain the claim and quarantine the
owner. These are logical trace claims, not bounds on process RSS or all CUDA
allocations. The one-use registry assumes a single trusted import of this
module in a cooperative owner; it is not protection against arbitrary code in
the same process. The separately owned model and forward temporaries require
their own independently justified, active commitment: a positive observation
estimate and its two trace claims do not establish that the workload fits.
The owner must obtain model and observation leases from the same actual
device-bound budget with verified telemetry, and must prevent early manual
release. The observer's lease type and active-membership checks cannot prove
those properties; process failure or restart needs outer accounting.

The CUDA entry checks CPython 3.11.15, PyTorch 2.8.0+cu128 / CUDA build 12.8,
Transformers 4.57.3, tokenizers 0.22.2, safetensors 0.8.0 and NumPy 2.4.6.
The pinned FLAN-T5-small configuration has eight encoder and eight decoder
layers, shared encoder/decoder input embeddings, and an output head with
separate storage. The observer checks the declared and actual block counts and
the embedding storage relationship; the later owner must still verify the full
checkpoint configuration and bytes.
It requires one intra-op and inter-op thread, deterministic algorithms without
warn-only mode, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disabled matmul/cuDNN TF32,
disabled cuDNN benchmarking, enabled cuDNN deterministic mode, disabled CPU
and CUDA autocast, and preserved CPU float32 subnormals. The owner must set the
environment before CUDA initialization and establish the verified runtime
binding before calling; the observer rechecks numerical state around guarded
operations. GPU arithmetic and denormal behavior are not asserted to match CPU
arithmetic. The trusted owner must also bind checkpoint bytes, model state,
seed, actual device identity, source code and fixed study inputs.

Each `output_live`, `captures_live`, and `device_refs_disposed` endpoint
synchronizes and supplies the recorder a snapshot of device-wide free/total
bytes and the PyTorch allocator's current and peak allocated/reserved bytes.
The recorder must persist the obtained record before applying its own
predeclared headroom policy; malformed counters fail the observation. Peak
scope is determined by the owner's prior reset and can cover more than this
forward. The final endpoint only concerns this forward's known references: it
does not dispose the model, establish a zero allocator, or prove post-process
device memory recovery. A full owner must separately sample pre-load and
post-cleanup memory, synchronize and verify disposal, preserve partial/failure
evidence, and require its supervisor's exact-child exit and cleanup evidence.

The caller retains a fresh `ObservationCustody` before entry. A failed owner
check, synchronization, hook cleanup or output validation leaves known live
device objects rooted there until their disposal phase. Each completed host
integer-bit copy is rooted in custody before its post-copy owner check and
subsequent validation; after the host result
has been built, `custody.host_result` also roots it through final synchronization
and on success. The owner must account for all of those roots before releasing
its observation claim. Successful device-reference disposal happens only after the
live work has been synchronized. A callback must return `None` on success and raise
on failure. An interrupted native return before Python assigns a reference,
unknown asynchronous work, or a wedged callback cannot be proven recoverable:
leave ownership uncertain and replace the supervised process. The checks are
sequential and require exclusive, unmodified model execution; they do not
isolate arbitrary concurrent changes. No profile may be promoted from a partial
or failed observation.
