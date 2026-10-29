# Proposed shared-service path for agreement experiments

This is a plan, not a measured profile, a frozen scientific protocol, or an
authorization to run held-out cases. The current DistilGPT2 development rows
are five tokens; the draft study permits unpadded batch-one inputs up to 32
tokens. The separate two-token shared-service smoke uses a different workload
and its measurements cannot authorize this one.

First add a fixed, bounded forward operation accepting one int64 token row and
its matching binary attention mask. Use the pinned DistilGPT2 checkpoint,
float32, eager evaluation and no cache. Return selected last-token scores, or
an explicitly bounded last-token vocabulary vector when the study needs it.
Validate token IDs, mask, dtype,
shape, requested output and maximum wire bytes before admission. The natural
study uses an all-ones mask, scores `is`, `are`, `was` and `were` (IDs 318,
389, 373 and 547), and retains bounded full last-token vocabulary logits for
specified controls. A broader binary-mask service requires its own matching
profiles. A fixed
request is owned by one authenticated team, with bounded pending and result
storage; reject overload fairly and make cancellation and disconnected-client
cleanup observable to the broker. Do not accept arbitrary user callbacks or
unbounded tensors.

Narrowing the wire result does not automatically narrow GPU work. A standard
32-token DistilGPT2 forward can first materialize `[1, 32, 50257]` float32
logits, 6,432,896 logical bytes, before selecting four scores. A dedicated
executor may instead apply the language-model head only to the final hidden
state, but it must be tested against the standard model on fixed development
inputs under the selected backend, with an explicit equality or tolerance rule.
Record any numerical differences; do not silently treat the two implementations
as identical. Measure GPU temporary and allocator peaks on the actual chosen
execution path and separately cap any CPU copies and serialized results.

Use separately approved resource keys for the actual sequence length, input
and mask shapes/dtypes, output selection, mode, exact source/model/runtime and
device identity, trace policy, and warmup state. Profile the real resident
worker and output path under the shared physical-device lease, with synchronized
memory samples and a predeclared safety margin; require live headroom in
addition to the estimate. A small development sample and a margin do not prove
that every input, allocation or external GPU process is bounded. Keep failed,
out-of-memory and cancelled attempts, and stop on uncertain drain.

After the forward-only path is reviewed, add a distinct ordered intervention
operation for explicit layer, module, invocation and local last-token
coordinates. The draft discovery scope includes attention output and MLP
output at layers 1–4, each 768 float32 values. The final-block subject-token
negative control uses a separate coordinate from the last-token candidates.
Capture and patch hooks require
one owner for the model, explicit pre/post-operation ordering and no client
callbacks. Enforce the draft's per-forward retained-tensor ceiling of 1 MiB and
the separately tracked 512 MiB study-wide ceiling, including source and
returned capture ownership. Profile each actual intervention/capture path and
its retained state separately; the forward-only estimate is not sufficient.

Before discovery, freeze the protocol, public development inputs, held-out
sampling rules, seeds, controls, source and model identity, result handling and
stopping rules. Freeze discovery predictions before drawing held-out rows. Do
not generate or inspect held-out rows merely to develop the service.
Use held-out intervention comparisons and controls to assess candidate
computations; activation traces alone are observations, not explanations.
Compare direct and gradient capture and run latency benchmarks only in their
own synchronized, isolated windows, separate from shared-load throughput.
