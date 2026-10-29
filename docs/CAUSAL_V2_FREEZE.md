# Synthetic v2 freeze and prediction builders

`looking_glass.causal_v2_freeze` builds canonical public pre-manifest and
discovery-freeze bytes, a private per-case prediction record, and its public
salted commitment. The code does not publish those bytes, run a model, generate
cases, enforce phase ordering or authorize a study. Its tests use fabricated
development records; they do not generate reserved v2 cases or model outputs.

The pre-manifest builder checks the preparation's exact public schema and
private commitments, fixed study seeds and split sizes in study mode, known
pilot and v1 inventory identities, a minimum source-file inventory,
clarification hashes, runtime fields, UTC timestamp and the frozen input-byte
identity contract. Source and runtime fields remain caller-supplied
attestations. A separately reviewed launcher and custodian must verify the
actual committed source, all executed and lazily imported project modules,
launcher/bootstrap/runner/scorer bytes, model-state hashes, runtime settings,
and current exclusions before publication. The builder's list is a minimum,
not proof that it includes every executed module or native dependency.

The runtime record also binds inter-op thread count, float32 matmul precision,
MKLDNN enabled/deterministic flags, deterministic-algorithm and warning flags,
CPU dispatch capability, and a bounded, exact six-key thread-environment map.
The launcher must inspect these settings from the actual process before
publication and verify they remain unchanged across execution. Recording a
value is not proof of its provenance or of native-code behavior. The minimum
inventory includes the runtime-preflight and v1-case sources plus the exact
pilot and current development-union files; their recorded hashes must match
the corresponding preparation commitments. A later union requires an
appropriately revised source binding before a real attempt.

`build_freeze` requires all ordered opaque cells and validates public sparse
traces, repeated clean observations, candidate/control outputs and reused
clean baselines across cells. It independently reconstructs each float32
logit contrast and patch effect from retained logits with integer bit
arithmetic, then recomputes first-change votes, mean L2 scores, selections and
polarity/constant/zero prediction means. Comparisons treat signed zeros as
equivalent where the protocol specifies. The resulting private prediction
entries follow model, cell, case, strength and site order. The public freeze
contains the prediction functions and discovery data, not held-out cases,
site assignments or concrete held-out predictions.

Before freezing, `verify_discovery_evidence(snapshot, cells, chunks)` accepts
the ordered, read-back **private** cell chunks and binds each sanitized
measurement to the saved float32 trace and logit bits, controls, case and
model identities. It checks injection arithmetic, target-capture and ordered
plan metadata, source/landing values, upstream and step-map controls through
the shared saved-control validator. Plan trace references must identify the
saved forward's run ID and site, and base references must use that run ID plus
`/logits`. Clean/repeat trace IDs must identify the same model and input when
reused; perturbed trace IDs must be distinct across discovery conditions. It
also checks reuse of the clean target across strengths. Signed
zeros are equivalent for the protocol's float32 comparisons; a nonzero
subnormal is not equal to zero. Failures use generic messages and these private
records, including their targets, never go to the selector. The helper checks
the records it receives; the custodian must verify they are the exact durably
published bytes and keep its inputs unchanged during validation.

These checks establish consistency of supplied records, not that a forward
occurred, that hidden-target injection integrity was checked, or that an
observed boundary uniquely explains behavior. `PreparedSnapshot` can be
constructed directly, so the custodian must call `verify_prepared` against
the exact retained exclusions and context rather than treating its type as a
provenance seal. The custodian must durably retain the private prediction
bytes and publish/read back the public freeze and commitment before any
held-out forward. After evaluation it must reveal the required records and
verify all commitments. These are cooperative process controls, not proof of
same-account secrecy or a trusted external timestamp.
