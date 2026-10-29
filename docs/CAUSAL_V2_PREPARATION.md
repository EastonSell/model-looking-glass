# Synthetic v2 preparation helpers

`looking_glass.causal_v2_preparation` implements the proposed v2 case formula,
collision ordering, separate salted held-out/schedule commitments, and a
private site permutation. It does not run a model, publish records, enforce
phase ordering, or authorize a study. Do not call `prepare_study` until the
source, model, storage, resource and study-execution gates have been reviewed
and are ready. Its tests use only different development seeds, fake secrets
and two-case fixtures; they do not generate any reserved v2 inputs or sites.

The production entry draws three distinct private 32-byte values with
`secrets`, checks the recorded original pilot digest and the canonical
projection of the 24 v1 cases, then derives the fixed six-model, 16-per-split
cases and a private nine-site permutation per model. It rejects overlapping
float32 tuples, treating signed zeros as equivalent for disjointness. It binds
the pilot, full caller-supplied development union, historical inventory and
commitment context into a public **fragment**, alongside salted commitments.
It cannot establish that the caller supplied every prior development input,
that random values remained secret, or that hashes describe executed code or
actual model states. The caller must verify those facts and bind the complete
manifest independently before any model work.

`Prepared` suppresses private records and secrets in its representation, but
its nested dictionaries remain mutable and its contents are not protected
from same-account processes. `verify_prepared` rebuilds the formula, checks
the exact context, case ordering, exclusions and schedule commitment, and
returns immutable canonical-byte snapshots. A later custodian must write and
verify those bytes durably in the required public/private order, freeze all
discovery selections and concrete predictions before held-out forwards, and
preserve incomplete or failed attempts. The helper itself is not a custody
gate or evidence that any phase was completed.
