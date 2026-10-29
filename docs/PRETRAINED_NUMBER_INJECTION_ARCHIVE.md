# Known-injection public discovery observations

`looking_glass.pretrained_number_archive` encodes a selector-facing discovery
observation in canonical JSON with versioned schema and base64-encoded original
little-endian float32 bytes. Each record binds its case, role, request, input
digest, 13 ordered boundary coordinates, action and donor coordinates, six
attention vectors, six MLP vectors, final-block vector and full last-token
logits. The per-record encoded cap is 512 KiB. Unknown fields, invalid roles,
noncanonical encodings, wrong tensor sizes, nonfinite bits, malformed typed
coordinates and incompatible action locations are rejected.

`audit_discovery_archive` accepts exactly 128 immutable records (eight cases,
16 roles each), reconstructs the site-blind selector cases and recomputes the
entire canonical discovery prediction file using the selector. It compares the
exact bytes, so the selector's per-payload hashes, control checks, votes,
margins and predictions must agree with the archived tensors. It returns a
domain-separated SHA-256 over the exact record hashes in case/role order; input
tuple order does not affect that digest. This is a consistency check using the
same selector, not an independent implementation of its numerical formulas.

This is a serialization and audit primitive: it writes or publishes nothing,
does not retain an interrupted attempt by itself, and cannot establish actual
execution, source identity, private add or self-patch landing, model integrity,
secrecy or publication chronology. The trusted custodian must store each
completed observation durably, preserve failures and available private
witnesses, check the latter separately, and publish the exact public records
before freezing predictions. Never include private injection metadata, nonce,
plan ordinals, owner tokens or full private witnesses in the public archive.
Same-process adversaries can access private data; this is procedural custody.

Tests use fabricated selector inputs only. They exercise all 128 records,
recompute the freeze, reject missing/duplicate/changed or noncanonical records
and nonfinite values, and check exact signed-zero/subnormal round trips and
private-sentinel exclusion. They do not run a checkpoint, allocate held-out
cases, or make a causal or performance claim.
