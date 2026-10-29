# Injection-only held-out split custody component

This component prepares one injection-only held-out split from already published
discovery artifacts. It does not generate a split on import, load a model, run
an intervention, publish a commitment, or prove publication order. No real
nonce, family, or held-out case is included in its tests.

The discovery source, runtime, plan, prediction, site commitment, checkpoint,
and protocol remain those published for the completed discovery. A separate,
prospective operational binding identifies the added custody, evaluator,
comparator, ledger and tokenizer sources. The caller must publish and review
that binding and any required deviation from the original same-source protocol
before drawing. The module checks that its own raw source hash matches the
binding; a trusted bootstrap must check those source bytes **before** import.

`prepare_study_once` is the real-study entry: it pins the already published
injection-only aggregate, discovery source, runtime, prediction, public plan
and site commitment before calling the generic `prepare_once` helper. It also
requires `mlg-injection-heldout-operational-binding-v2`, which separately
identifies the preparation child, structural auditor and outer observer. The
generic helper retains v1 compatibility for historical or fabricated inputs;
it must not be substituted for the fixed-study gate in production.

The helper accepts immutable public bytes, a private site nonce and record,
the protected parent path, and trusted publication, exposure-observation,
tokenizer and ledger callbacks. The caller must independently verify the exact
published bytes and their observed ordering, the current reviewed inventory,
the pinned tokenizer assets and runtime, the published operational sources,
the shared ledger's reconciled state, the common gate root and source, and the
private parent's access controls.
The callback interface is cooperative: a checksum or `True` return value does
not prove these external steps or protect against another process with the same
operating-system account. The reviewed discovery request plan was separately
audited; this component binds its exact published bytes and entropy but does not
recompute its full request schedule. The operational binding lists exact hashes
for the supplement, owner, executor, journal, supervisor, auditor, shared gate,
lease, comparator, custody, ledger, tokenizer and accounting amendment, plus the
inventory source revision/hash and inventory digest. V2 also binds exact
`preparation_child_source_sha256`, `preparation_audit_source_sha256` and
`outer_observer_source_sha256`; the existing auditor field refers to the
independent held-out numerical auditor. A trusted `source_check`
must check those published and locally executed bytes, and a trusted
`prediction_check` must validate the frozen prediction's schema and means using
the prebound discovery scoring source before either claim or nonce draw.

Preparation checks the public freeze and the existing site commitment before
claiming `injection_heldout`. It checks the phase's fixed 72 MiB logical and
192 MiB physical shares, writes a claim record, announces the allocation under
the shared gate, rechecks current exposure and source, and draws one 32-byte
nonce from `secrets.token_bytes`. The nonce must differ from both the private
site nonce and the public discovery-plan entropy. It checks actual tokenizer
output for every family member, one-token subject alignment, shared context,
and all four single-token continuations. It rejects a whole-row prefix in either
direction against the selected known inventory, in addition to the frozen
sampler's own split and exposure checks. A collision or other failure stops the
one-use attempt; no replacement nonce or case is drawn.

The protected phase directory retains `claim.json`, `nonce.bin`, the canonical
`heldout-record.json`, exact `inventory.json`, public `commitment.json`, and
`preparation-complete.json`. The latter binds the commitment file and the
operational binding; failures retain `preparation-failure.json` when possible.
The ledger claim remains consumed even when an interruption prevents a later
file from being created. The returned object contains only the private phase
path and the public domain-separated commitment digest. The caller must publish
and independently read back the exact commitment file before any held-out model
forward and must retain all partial files and reconcile their physical sizes.
Private nonce, record, site, tokenizer diagnostics and exception details must
not be sent to an untrusted selector, public log, filename or terminal. A
supervised entry point must sanitize all uncaught `BaseException` output.

For separate natural and injection splits, `compare_committed_splits` verifies
both canonical private records against their independently observed public
freezes and commitment files, using the injection site tuple only inside the
trusted custodian. It compares every full token row in both prefix directions
and returns a canonical `mlg-number-disjoint-split-check-v1` receipt containing
only the two freeze hashes, domain and file commitment hashes, inventory
hashes, comparator source hash, two integer case counts, and `clear` or
`collision`. It returns no text, token IDs, nonce, site or private path. If
both private splits already exist before an affected evaluation, the relevant
owners must observe and bind the same verified clear receipt before forwarding.
If one track completes independently first, its revealed rows become known
exposure for the later fixed attempt; a collision stops that later attempt
without a redraw and does not invalidate the completed result. An unallocated
other split is not a clear comparison. `verify_clear_receipt` checks the exact
schema and integer counts, both public commitment files and expected bindings;
the receipt remains a trusted custodian report, not independent proof of
confidentiality or timing.

This library is not an end-to-end study launcher. Before using it for a real
draw, the separately reviewed owner must bind a source-first supervised entry
point, the actual v6 reconciled ledger implementation and phase grant, official
publication readbacks, current exposure observation, tokenizer/checkpoint asset
verification, source and runtime checks, failure/descendant cleanup, private
ACLs, and the split-comparison handoff. The evaluator separately verifies all
commitments and frozen predictions and keeps the concealed site unavailable to
the selector. Neither known-exposure checks nor the public sampling frame prove
unknown historical or pretraining independence.
