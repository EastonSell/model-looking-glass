# Development-only synthetic intervention executor

`looking_glass.causal_v2_executor` exercises the proposed v2 controls on the
two retained reviewer inputs with first coordinates `+2` and `-2`. It rejects
other input vectors; these two tuples are already in the published v2
development exclusion union. This helper does not generate a discovery or
held-out split, create a site schedule, enforce custody or phase order, seal
predictions, publish evidence, or authorize a study. It is an instrumentation
test for a small artificial residual model, not an explanation of a pretrained
model or evidence of its reasoning.

For a caller-supplied evaluation-mode `TinyResidual(width=8, depth=4)`, one
retained development vector, one of the nine specified artificial targets,
and strength 0.25 or 0.75, the helper uses 19 root forwards: a clean sparse
trace and repeat, a private clean target capture, then the 16-forward active
condition. It validates both private capture locations and retained float32
bits, confirms the native injection, and checks every sparse output before the
known injected block privately for equality. A separate sparse comparison
reports the first observed boundary; an observation is not a causal origin.
This per-condition helper recomputes the clean traces and target capture for
each strength. The real v2 schedule reuses those results, so invoking this
helper unchanged for all conditions would exceed its planned forward count.

Each active condition includes three clean-to-perturbed patches, three reverse
patches, three nulls, three actual same-forward self-patches, an upstream
negative control, and a final clean check. Every patch/null uses an exclusive
ordered `ActivationPlan` with an immediate post-action capture. The self-patch
captures its own current-forward source, applies that source, and captures the
result. The helper verifies full logits, finite float32 source/landing bit
patterns with signed zeros treated as equal, ordered
metadata, whole-output coordinates, invocation zero, capture bytes and source
bytes. It retains the original bit patterns and returns all 13 plans' raw metadata, plus a
`DiscoveryCondition` that carries only the candidate observations/logits.
Ground-truth target and target-capture metadata remain in a separate private
integrity object; the caller must keep that object away from the selector.
Case, model, source and base identities must be bound by a future custodian.

The helper requires the checked v2 numeric policy, CPU float32, one intra-op
thread, no autograd during each forward, and a model tree without existing
local/global forward or backward hooks or aliased module paths. Its budget counts a
submission before hook preparation and a root call immediately before the
model is invoked. These checks rely on cooperative in-process execution; they
do not prove that another thread or noncooperating code cannot mutate a model
or install a hook. The helper is not the protocol-global fail-stop mechanism.
After any failure or cancellation, the future runner must record an incomplete
attempt and discard or quarantine its model, budget and worker as appropriate,
especially when hook cleanup is uncertain. Returned in-memory data must be
durably saved by the caller; this helper alone does not persist failures or
meet the 1 GiB host-disk evidence budget.
