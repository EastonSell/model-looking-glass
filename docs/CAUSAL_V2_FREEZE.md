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
