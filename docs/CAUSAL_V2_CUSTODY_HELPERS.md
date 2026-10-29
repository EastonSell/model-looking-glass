# Synthetic v2 custody: pure evidence helpers

`looking_glass.causal_v2_custody` provides small, framework-independent helpers for
the proposed synthetic-v2 custody workflow. It does **not** generate a split or
site schedule, establish a phase gate, publish a commitment, execute a model,
or authorize a study. No real v2 discovery or held-out data are needed by its
tests. A future runner and independently reviewed custody gate must enforce the
actual protocol and its pre-seal clarifications before any study execution.

`build_exclusion_union` merges caller-supplied records containing exact
float32 values, rejects duplicate IDs and tuples (treating signed zeros as
equivalent), sorts by ID and returns the protocol's compact canonical UTF-8
JSON. Its integer IEEE-754 conversion checks whether each supplied Python
number is exactly representable as binary32 without relying on an ambient
float32 cast; it therefore distinguishes subnormal values even when the CPU
is configured to flush them. The union does not establish that every prior
development input has been supplied.

`verify_split_disjointness` checks caller-supplied **projections** consisting
only of `case_id` and eight `values` against the canonical development union,
each other and a required caller-supplied historical list. It does not prove
the historical list is complete, the input rows are authentic, or the lists
have the protocol's required size, order, seeds, indices or split identities.
The future gate must validate those full records and derive projections from
the exact records it commits; a successful disjointness check alone is not a
valid v2 split.

`commitment` and `verify_commitment` implement the v2 domain separators,
32-byte nonce and canonical JSON formulas. They validate the outer record
keys, binding-hash format and JSON value types, **not** the nested record's
schema, completeness, ordering or consistency with the manifest. A matching
digest verifies those supplied bytes only; it does not prove randomness,
secrecy, an external timestamp or execution. `validate_nonce_set` checks that
the three byte strings are distinct, correctly sized and not the public
discovery root; it does not sample them or prove their entropy. Real custody
must itself draw the specified secrets and independently validate and retain
the complete committed records.
