# Synthetic causal v2: pre-seal patch and null delivery clarification

This clarification supplements the proposed, pre-seal
[v2 protocol](CAUSAL_CONFIRMATION_V2.md) and the
[numeric](CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md) and
[source-execution](CAUSAL_CONFIRMATION_V2_SOURCE_CLARIFICATION.md)
clarifications. Commit and independently review it before any v2 split or
site schedule is sealed. Bind its SHA-256 in a separate top-level
`self_patch_clarification_sha256` field of the public pre-manifest, without
changing the prescribed exact keys of the commitment `context` object.

The proposed protocol calls for a same-run self-patch but also says that no
composite hook is assumed. The independently reviewed
[`ActivationPlan`](INTERVENTION_PLANS.md) now supplies the missing capability.
For each of the three candidate block-output sites in every active condition,
perform one perturbed-base forward with one exclusive plan that contains
exactly these ordered operations at that same module, invocation zero, and
whole-output coordinates:

1. `capture` a detached snapshot named `self_source` of the current output.
2. `patch` that same output using `source_capture="self_source"`.
3. `capture` the output after that patch as `post_action`.

Use native `TinyResidual` perturbation, `max_tensor_bytes=32`,
`max_capture_bytes=64`, `max_source_bytes=32`, `max_steps=3`, and
`max_invocations=1`. Do not nest the plan with any independent trace or
intervention session. No value from an earlier forward may be used as this
self-patch source. Execute under the protocol's CPU, thread, `torch.no_grad()`,
and checked denormal settings. The resulting full logits must match the
unmodified perturbed-base logits from the same model/input/configuration and
strength under the numeric clarification's finite, bit-pattern equality with
signed zeros equivalent. Save the full control logits, base logits and
source/base identity as required by the original protocol. A failure or
missing capture invalidates the cell and is retained; a self-patch is not a
substitute for the separate null, upstream, clean, or downstream controls.

After successful plan exit, verify and retain sanitized plan metadata: capture,
patch, then post-action capture all refer to the intended candidate module,
invocation zero, whole-output coordinates and selector; selected/output shapes
are `[1,8]`, CPU float32; the patch names `self_source`; capture bytes are 64
and copied external source bytes are zero. Retain both detached captures
losslessly. Require finite bit-pattern equality (signed zeros equivalent)
between `self_source` and the same-condition perturbed sparse-trace value, and
between `post_action` and `self_source`. The first is a separate-forward
consistency check; the patch source is still the current forward's capture.

For each of the three clean-to-perturbed, three perturbed-to-clean reverse,
and one upstream patch forwards, use one exclusive plan at the intended site:
`patch` from the protocol's same-input clean or perturbed source, immediately
followed by `capture` of `post_action`. Bound the full tensor, aggregate
capture and external source to 32 bytes each, with `max_steps=2` and
`max_invocations=1`. Require `post_action` to equal the intended source under
the same finite bit-pattern rule. The comparison source must be an independently
owned snapshot of the specified clean or perturbed sparse trace, with its
identity bound and integrity checked across plan construction and execution;
do not compare only with a mutable argument passed to the plan. For each of the three null forwards, use one
exclusive two-step `null` then `capture` plan with the same 32-byte caps and
invocation limit; require `post_action` to equal the same-condition perturbed
sparse-trace value. Check and retain sanitized metadata for actual order,
site, whole-output selector, invocation, CPU float32 shape, and expected
capture/source byte counts (32/32 for external patches and 32/0 for nulls),
plus the detached post-action values and the
referenced source or trace identities. These state checks supplement all
original full-logit checks; an unchanged logit alone does not prove a patch
landed. A metadata, value, or required output mismatch invalidates the cell.
These captures verify the value immediately after the plan action at its hook;
the executor must ensure independent local/global hooks and custom call paths
cannot alter or bypass it. The plan does not police such noncooperating hooks.
Continue the original final-logit and post-hook clean checks: a landing capture
alone does not establish the downstream behavior.

Keep hidden target-identifying diagnostics private until reveal. Held-out
post-action/self-source captures and control logits are held-out outcomes;
retain them in the private custodian record until evaluation finishes and the
protocol permits reveal, even when metadata does not identify the hidden site.
Include all serialized captures, source references, metadata and other artifacts
in the [1 GiB host-disk evidence budget](CAUSAL_CONFIRMATION_V2_RESOURCE_CLARIFICATION.md),
which supersedes the earlier 64 MiB proposal and is separate from GPU VRAM.
Retain complete raw evidence when it fits, with space reserved for completion
and failure records. Sparse traces and private integrity captures remain separate
exclusive sessions. The three self-patches replace the three identity-patch
forwards already counted in each condition, so the planned root-forward count
remains **57,408** and the cap remains **60,000**. The ordered plan's additional
copies, hooks and validation do not alter that arithmetic but may change time
or peak memory. Before sealing, test the exact control and failure cleanup on
development-only models and inputs, and assess resources without using any v2
held-out cases. The 16 retained 32-byte snapshots per active condition imply
1,769,472 raw tensor bytes over 3,456 conditions, before metadata, source
references, serialization and all other evidence. Check the actual output
layout against the host-disk budget; raw tensor bytes alone do not establish
fit. Any added development vectors must join the exclusion inventory
for both future splits. Before preparation, construct and publish a canonical
UTF-8 compact JSON union record `{"inputs":[{"case_id":...,"values":[...]},...]}`
sorted by case ID, with unique IDs and unique float32 tuples treating signed
zeros as equivalent. It contains the unchanged original pilot tuples and every
other retained development tuple, including
[`review-dev-41977`](../results/synthetic-v2-dev-review-41977/). Bind its
SHA-256 as a top-level `development_exclusions_sha256` in the pre-manifest;
retain the original pilot digest separately and reject overlap with the full
union in **both** splits. If the inventory changes, recommit before preparing
either split. The recorded 202-forward pilot used a prior-forward same-input
identity source and did not measure the same-forward self-patch or post-action
capture plans; its original evidence remains unchanged.
