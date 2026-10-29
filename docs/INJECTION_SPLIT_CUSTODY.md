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
requires `mlg-injection-heldout-operational-binding-v3`, which separately
identifies the preparation child, structural auditor, outer observer, the
source-first gate callers, the prospectively published runtime deviation and
the observed held-out runtime. The generic helper retains v1 and v2
compatibility for historical or fabricated inputs; it must not be substituted
for the fixed-study gate in production.

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
`outer_observer_source_sha256`, plus `gate_allocation_caller_source_sha256`
and `gate_transition_caller_source_sha256`; the existing auditor field refers
to the independent held-out numerical auditor. A trusted `source_check`
must check those published and locally executed bytes, and a trusted
`prediction_check` must validate the frozen prediction's schema and means using
the prebound discovery scoring source before either claim or nonce draw.
V3 binds exact `runtime_deviation_sha256` and `heldout_runtime_sha256` bytes;
it also binds the separately published compatibility-replay plan, the replay
producer, claimant, supervised owner, read-only preclaim-guard and phase-accounting sources, and
the completed replay evidence. A trusted source check must verify the actual
published and executed bytes for each of those identities.
The caller must independently verify the observed current runtime and replay
against the prospectively published requirements. The original discovery runtime
digest remains unchanged.

The separately supervised claimant must verify the frozen public inputs and the
published prospective replay plan before consuming the single ledger claim. The
plan binds both runtime hashes, the fixed roles, public references, tolerance,
replay source and the diagnostic and replay budgets. After that plan is
published, the owner makes the single claim and runs the fixed public replay.
The claim and every partial or failed replay remain consumed. Once successful
completion and cleanup are independently checked, the owner publishes the replay
result and the complete operation binding and observes their exact bytes.
Preparation may begin only after that readback; its source-bound caller must verify the exact
protected ledger row and phase path, original claim history, and all phase-side
evidence rather than pass a cached or caller-created grant. The fixed study entry
uses `prepare_study_preclaimed`: it writes an exclusive durable start marker,
checks that real ledger row and reconciles side evidence through trusted callbacks,
and never makes a second claim. The reconciliation callback receives explicit
`before_phase_creation` and `before_nonce` stages; it must verify an absent phase
at the former and the expected claim-only phase at the latter. A failed or interrupted handoff must be retained
and cannot be retried, including a failure before the marker can be written.
The helper latches a start attempt in the current process before marker creation,
so a failed open cannot be retried there. The separately supervised owner must
also durably record its one-use preparation launch before starting that process
and refuse every restart after a failed or uncertain launch; an in-memory latch
cannot protect a new process if marker creation failed before creating a file.
The 72 MiB logical and 192 MiB physical phase limits include the fixed 8 MiB
logical and 16 MiB physical replay reservations, plus 4 MiB physical for the
retained no-model rehearsals; all partial, failed and duplicate side evidence
counts. The external owner must verify actual file sizes, free space, source,
published ordering and private path identities. The marker binds the exact
ledger and phase paths, configuration, operation, replay-plan and replay-evidence digests;
it does not itself prove that a callback observed the real ledger or evidence.

Preparation checks the public freeze and the existing site commitment, checks
the fixed phase limits, writes a claim record, announces the allocation under
the shared gate, rechecks current exposure, runtime, source, the protected claim,
side evidence and unchanged handoff, and draws one 32-byte
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

The final ledger check, nonce call and durable nonce write run while the source-bound owner holds the
reviewed guard's short SQLite `BEGIN IMMEDIATE` reservation. The gate lease must
already be released, and the outer watchdog bounds the local checks and write;
no network work is performed under that reservation. The owner must also prevent
out-of-band database replacement. An uncertain lock release after entropy access
is incomplete and cannot be retried, even if interruption prevented retaining
the OS-returned bytes. The exact canonical claim record and handoff marker are
rechecked at this boundary; a change after entropy access leaves the retained
nonce and stops the attempt.
