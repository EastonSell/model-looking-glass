# Synthetic v2 runner components under development

These components support the proposed [synthetic v2 protocol](CAUSAL_CONFIRMATION_V2.md).
They are **not a runnable or sealed study** and do not generate cases, produce a
site schedule, run a model, publish commitments, or evaluate held-out data.
The private pilot-input exclusion record, numeric clarification, source
bootstrap, custody and execution gates must be bound and independently reviewed
before any real v2 split or site schedule is prepared.

`causal_v2_runtime.establish_numeric_policy()` is for a **fresh isolated CPU
execution process**. It requests denormal preservation, checks input and output
subnormal probes before and after setting one PyTorch intra-op thread, and
returns a policy bound to that process and execution thread. It changes the
thread's denormal mode and the process-wide PyTorch thread setting; it does not
restore them and must not be run in an application or shared worker process.
The future runner must verify the policy before and after model work and
relevant comparisons, and retain failures. The record contains the actual
requested flush mode, request success, probe results, thread policy, process,
architecture, and PyTorch version/build. A successful probe does not attest to
other threads, arbitrary external libraries, source provenance, or previous
model computations. Direct `NumericPolicy` construction is guarded against
accidental use; Python introspection or mutation means this is not a security
boundary.

`float32_exact_equal` compares finite, dense CPU float32 snapshots of at most
eight elements by their saved bits, treating signed zeros as equal. Shapes must
match and empty, nonfinite, named, nested and lazy negative/conjugate tensors
are rejected. It neither establishes the model's execution mode nor verifies
trace coordinates or model/input identity.

`causal_v2_selection.freeze_discovery` takes exactly 16 ordered discovery
records, each with two strengths in protocol order, sparse clean/repeat and
perturbed traces, complete logits, and the three patch/control sets. It checks
the sparse event policy, repeated-clean equality, upstream negative, late
downstream rescue, null, identity, and post-hook clean controls. A difference
in the upstream sparse block invalidates the cell; an active observation miss
is counted separately from cell abstention. The first-changed vote and mean-L2
ranking tie toward the lower eligible block; all-zero observations abstain. It
fits polarity-conditioned, constant-mean, and zero effect predictions for all
three eligible sites in the prescribed case/strength order. The uniform random
baseline is recorded as the exact distribution `1/3` per site, not as a
best-of-three choice. The returned JSON-compatible discovery observations omit
run IDs and do not accept an injection target or a held-out outcome.

The selection function requires an active numeric policy and explicit caller
attestations that captures are complete, paired on the intended model/input,
polarity is correct, patch sources match that input, and private
target-identifying integrity and step-map controls succeeded. Those attestations
are not independently verified by this function and do not establish blindness,
provenance or phase order. The caller must preserve the snapshots during the
call, attach the correct public case and opaque cell identities, durably retain
sanitized failure records, and enforce all remaining protocol controls and
resource budgets. No empirical outcome is claimed by these development tests.

On a valid cell, the no-injection summary explicitly records zero false
detections and abstention by both aggregate selectors. If a clean repeat
differs, the function raises `DiscoveryInvalid` with sanitized partial
`no_injection_failure` evidence, including the trace comparison and full clean
and repeated logits; it does not return a valid cell. The future custodian
must retain that partial evidence, complete the run-level no-injection report
where possible, and deduplicate the 16 shared model/input negative controls
across configurations and strengths rather than counting them repeatedly.
Stopping after the first failure leaves that inventory incomplete; the retained
failure is not by itself a complete 16-case false-detection rate.
The exact implementation of the protocol's same-run self-patch control is
still awaiting clarification; the selector merely validates supplied logits.
