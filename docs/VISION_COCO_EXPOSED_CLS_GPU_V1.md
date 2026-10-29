# Exposed-pair CLS versus spatial ViT CUDA exploration

This prospective exploratory run uses one previously exposed v4 pair. It is
not the v5 evaluation, a fresh held-out test, a population estimate, or a
CPU/GPU bit-equality test. The existing CPU v5 study continues independently.
The pair was selected here solely because it is first in the published v4
selection order; its v4 spatial outcomes were already known. Freeze and publish
this protocol and the exact source before any forward in this experiment.

## Fixed inputs and prior information

Use pair-00 from `results/vision-coco-recognized-v4-20261029/screen-result.json`
(SHA-256 `711894680eac6bed8e60b8acb4413775cce51f5630115e561ecbfe0504efa648`)
and the published v4 manifest (SHA-256
`21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb`). The
fixed target IDs are cat 405058 and dog 196065; their same-species donor IDs
are 69231 and 142769. The common spatial block is tokens `[90,91,104,105]`.
Verify all four JPEG hashes and processed-pixel hashes against the manifest,
the fixed model/processor assets, and the checkpoint state before forwarding.
The legacy `heldout` field in the saved pair is a schema label, not evidence
that these images or this comparison are held out.

The prior v4 spatial-only CPU experiment reported opposite-donor oriented
changes of approximately 0.00258 for the cat and 0.03834 for the dog, and
the pair did not pass its earlier composite. Its donor selection and outputs
were already observed. Its q used spatial norms and distances alone. The
present common CLS/spatial q and fixed random signs differ, so its spatial arm
is not an exact replication of that CPU intervention. These inputs may also
have appeared in model pretraining; no causal population or localization claim
follows from one selected pair.

## Fixed scientific actions

Use the pinned ViT-tiny 224×224, batch-one, 1,000-class checkpoint in eval,
float32 and eager attention. Obtain fresh same-GPU clean logits and block-2
and final-block activations for each of the four fixed images. The selected
targets' GPU baselines must satisfy the original sign, gap and species-mass
checks; otherwise save the incomplete attempt and stop without replacing a
case. Record each donor's same-GPU top-one class and cat/dog group masses;
the donor recognition and selection were from the prior CPU screen and are
not a new GPU recognition gate. Compare the saved CPU and GPU clean logits descriptively using their
full vectors and max/mean absolute differences. Do not require CPU/GPU bits to
match or select a new pair based on the differences.

Use encoder block index 2 (the third block), CLS token 0 and the fixed four
spatial tokens. For each target side, let q be the minimum of 0.099 times
each site's receiver norm and the distances to both its opposite-species and
same-species sources at both sites. Calculate all distances, donorward
interpolations and the fixed `vision-coco-cls-v5:{pair_id}:{side}:{site}:{i}`
random signs in host binary64 with nearest-even rounding, then cast each
replacement once to float32. Check rounding and float32 denormal preservation
at the execution boundaries; these point checks cannot prove the mode did not
change and recover within an unchecked native call.
Record q, both receiver norms, four donor distances, six delivered norms and
their receiver-relative fractions. Do not extrapolate donors. Apply the
published `min(max(1e-5,1e-5*q),0.01*q)` dose tolerance. Each of the six
realized doses must also be within 1% of that side's opposite-CLS realized
dose. Retain dose-ineligible/unmatchable outcomes without adapting q.

For each side perform clean repeat; null and self patches at both sites;
full-state self and opposite-donor recovery; the opposite, same-species and
fixed random matched interventions at both sites when dose-matchable; a final
layer non-CLS control; and post-clean. Four initial clean calls plus at most 15
calls for each side give at most 34 root forwards. Freeze the deterministic
conditional schedule and its dose-status branches before model work; record
the branch selected for each side when its clean activations determine q.
No failed or uncertain forward is retried. All clean/repeat, null/self and
specified full-recovery equality controls are exact within this GPU owner;
every patched forward retains same-invocation before/after witnesses and
requires exact float32 landing and bitwise unchanged unselected tokens.

Retain all 1,000 finite logits, full required before/after states, selected
witnesses, requested values and per-action metadata. Compute the dog-minus-cat
log-sum-exp score, full-distribution total variation and KL(clean || patched), signed and
annotation-oriented movements, and the CLS/spatial control-adjusted advantages
using the reviewed scientific primitives. Report both sides and every control,
including negative values, dose exclusions and failures. The v5 per-side
thresholds may be shown descriptively, but one exposed pair cannot support
the v5 aggregate or a confirmatory success claim.

## CUDA custody and bounded evidence

Use a reviewed source-bound direct child and the existing canonical physical
GPU lease and sticky-marker supervisor, not a new general shared service. Bind
the exact scientific runner, CUDA adapter, model assets, input/reference bytes,
framework/runtime and protocol source before execution. Use one host thread,
deterministic algorithms, native PyTorch allocator, disabled TF32 and autocast,
and a float32 default dtype. Check that every module remains in evaluation
mode at the scientific boundaries; this is not a continuous state guarantee.
Record actual GPU UUID, driver, device name/capability and dependency
versions. Admit a 25%-of-device PyTorch cap plus 256 MiB commitment with the
existing `max(1 GiB, 15% total)` free reserve, checked on fresh CUDA memory
points under the lease. These are cooperative point checks, not a reservation
against unrelated GPU applications or a bound on non-PyTorch CUDA allocations.

Declare at most 16 MiB of simultaneous GPU trace and 16 MiB of simultaneous
host trace custody and at most 1 MiB of retained tensor evidence per forward.
Unlike the earlier four-token pilot, a full-state source is 151,296 bytes;
two full before/after captures are 302,592 bytes, so those three alone overlap
at 453,888 bytes before output, selected witnesses, temporary or copied values.
The hook can also hold the original full layer output, the previous patched
full output and a new patched full output simultaneously; source clones,
input copies and captured values are additional. The returned model output
must have hidden-state and attention outputs explicitly disabled. The CPU
runner can retain the preceding result while the next call runs, its initial
states and the planned replacements; the writer can hold tensor copies, byte
pieces and the joined payload concurrently. Count these overlaps separately.
Set and test an adequate source limit instead of reusing the pilot's 3,072-byte
limit. Four float32 inputs occupy 2,408,448 bytes, and four pairs of initial
early/final states occupy 1,210,368 bytes before any other host copies. These
inventories do not constitute an RSS or CUDA-context bound. The exact root
inventory and failure-path custody must be reviewed against the frozen source.

Synchronize after each forward while model, input, plan, output, source and
capture roots remain held. Copy bounded evidence to host and persist it before
validating/retiring that forward's roots; preserve the fixed small clean-state
inventory needed for the remaining actions. Record fresh free/total and process
allocated/reserved values and cumulative allocator peaks through all actions.
Guard each top-level CUDA transaction before and after; an ownership loss
during a native call can be noticed only at its next check. On OOM, uncertain
completion, interrupted owner check or evidence failure, retain known roots,
keep the partial attempt and unsafe marker, make no salvage CUDA calls and do
not retry. On success, use the reviewed synchronized model/tensor disposal and
allocator/workspace cleanup, retain the physical lease through child exit, and
independently record its reap and marker state. Local zero counters do not
prove global device idleness or native destruction.

At admission require at least 3 GiB free on the output volume: a 1 GiB bound
for the planned attempt and its temporary evidence, plus 2 GiB remaining
headroom. Also require 1 GiB free on the canonical workspace volume for the
lease, marker, existing assets and residual runtime writes. If these are the
same volume, require at least 4 GiB there. Place controller temporary files,
owner temporary files, Python bytecode, and Hugging Face and CUDA caches on
the output volume; keep the source, existing checkpoint and physical lease in
the workspace. Native runtime writes may still use other locations. The 34
forward payloads are capped at 1 MiB each; controller logs and dependency
output are not subject to an operating-system quota. These free-space points
are not reservations against other writers. Record per-stage process working set, peak working set
and private bytes alongside the GPU points; these do not establish a bound on
total host memory. Saved evidence is closed before its roots are retired;
without an explicit fsync it is not a power-loss durability claim.
Publication must retain exact input/source/runtime
identities, all outputs and controls, incomplete attempts, the direct-child
receipt and an independent saved-byte scientific replay. State explicitly that
the local source and process checks assume trusted dependencies and do not
authenticate every executed instruction or account for unobserved descendants.
