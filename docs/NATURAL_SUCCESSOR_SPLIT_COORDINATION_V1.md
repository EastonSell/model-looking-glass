# Natural successor split coordination, v1

The prospective natural successor has a separate phase, root and markers. The
original natural claim, commitment, split and gate markers remain untouched;
its first-forward announcement is a possible exposure. This document defines
a cooperative ordering interface, not a private-comparison result or a grant to
run a model.

Both the successor and injection owners must use the existing shared gate
`work/agreement-study-ledger-v5/cross-track-gate-v1/gate.lock`: gate-directory
identity `(1219152768, 4503599632054271)` and persistent lock identity
`(1219152768, 6192449492328171)`. The successor markers live separately in
`work/natural-agreement-successor-v1-20261029`, directory identity
`(1219152768, 1125899911728318)`. The calling operator must independently
verify both roots' current protection, path and filesystem identities, verify
all exact published source bytes before importing them, and hold the actual
shared lock while reading and updating these states. The identity of a lock
alone is not proof of its current ownership.

`looking_glass/natural_successor_split_state.py` checks the held, verified gate,
fixed paths and identities, and canonical bounded marker schemas. Its
`inspect(gate, root)` returns a digest-only
`mlg-natural-successor-state-disposition-v1` object:

* `not_allocated_unchecked`: no phase or allocation is recorded. It is not a
  disjointness result.
* `allocation_pending_unchecked`: the separately claimed phase has an
  allocation marker but no public commitment. This is not a disjointness
  result.
* `committed_comparison_required`: an allocation and public commitment have
  been announced; the result binds their marker and public commitment digests.
* `possibly_exposed_comparison_required`: a first-forward announcement also
  exists; the result binds its digest and its original unchecked peer status.

An orphan phase claim, owner-start marker without its forward announcement,
owner directory without a commitment, unknown or malformed marker, missing or
changed root, invalid lock, or legacy query artifact causes a stop; it is never
an absent or clear state. The transition, query, phase-claim, owner-start,
nonce, heldout-record and commitment filename prefixes are reserved: unknown
or partial names in these namespaces also stop. Successful `inspect` output is a point-in-time observation and
must not be used after releasing the lock without a fresh check. The injection
owner must call this exact inspected interface while holding the same lock,
retain a canonical digest-only disposition in its own supervised operation,
and keep the lock through its injection forward announcement. That retained
record binds the inspected output, the source/operation and its own injection
commitment and forward marker identities, so it documents their ordering. The
helper does not itself write an injection disposition or observe its caller's
process exit.

Before creating the separate phase claim, the successor owner holds the shared
lock, establishes exclusive cooperative control of its root and verifies no
legacy query process is in flight. It creates the one-use `phase-claim.json`
and calls `announce_allocation(gate, root)` before releasing the lock or drawing
the nonce. The canonical allocation marker binds the phase claim, operation,
frozen prediction, pool, full inputs, public-support receipt, amendment and
selected inventory digests. A crash between claim and announcement leaves an
unannounced claim and requires reconciliation; it is not safe to retry. File
flush and readback do not guarantee directory-entry durability after power
loss. An interrupted or unobserved result stays consumed/uncertain.

After the nonce has selected the separately committed split, the canonical
public commitment has exactly these fields: `schema` with value
`mlg-natural-successor-public-commitment-v1`, `prediction_freeze_sha256`,
`pool_sha256`, `inputs_sha256`, `pool_clearance_sha256`,
`exposure_inventory_sha256`, `old_natural_commitment`, `protocol_sha256`,
`amendment_sha256`, `operation_sha256` and `heldout_commitment`. The last is
the salted record commitment; the nonce, private rows and unsalted record hash
are not published. A source-bound operator independently verifies the exact
public bytes on the intended official main history, then calls
`announce_commitment` while holding the gate. Its publication callback is an
operator trust boundary: mere callback success does not prove official-main
ancestry or a correct private preparation.

The successor's first-forward transition permits only an absent or allocated
but uncommitted injection peer, recording the corresponding explicit
`injection_not_allocated_unchecked` or `injection_allocation_pending_unchecked`
marker. A committed injection peer stops this version of the helper: a later
separately source-bound, independently observed pairwise comparison and
reviewed receipt integration are needed before an affected forward. A prior
injection first-forward announcement is treated as possibly exposed. The
successor operator must also re-observe the current official exposure inventory
at the actual first model call, hold the gate through that call's first return,
and complete the protocol's post-run checks.

The transition helper is not owner-run authority. In particular an `owner/`
tree that appears after commitment may be startup scaffolding or an interrupted
owner; the natural owner must enforce its own fresh one-use start/output
boundary and may not reuse an existing tree merely because this helper still
reports a committed state.

Conversely, if the successor is committed or possibly exposed before a later
injection forward, the injection owner must separately compare the two
committed splits using a reviewed source-bound operator, observe its successful
exit and receipt, and recheck both commitment identities and that receipt under
the shared lock. The earlier original-natural/injection comparison remains a
separate requirement. A public-looking receipt or an arbitrary success callback
alone is not authentication. Comparisons run outside the short lock; their
exact identities are rechecked inside it. Neither track is delayed solely by
an absent or still-uncommitted peer, but the later affected track carries its
own comparison obligation.

These markers and checks are cooperative. They do not prevent same-account
tampering, establish an independently witnessed process exit, or lock arbitrary
unrelated model users. The existing scientific protocol, separate phase budget,
source custody, official currentness and complete no-model production-lineage
rehearsal remain prerequisites before any draw or model forward.
