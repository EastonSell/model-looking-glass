# Agreement-study phase accounting

This prospective clarification applies to the pretrained number/agreement
study after its existing natural discovery attempt. The frozen protocol's
512 MiB limit concerns **unique retained tensor payloads**, including the
3,216,448-byte public preflight logits. Its separately planned 1 GiB allowance
concerns study evidence on host storage: retained originals, private and
public copies, staging, receipts, metadata, failed attempts and orphans. A
serialized copy consumes physical allowance even when its tensor payload was
already counted logically. These are neither VRAM limits nor estimates of peak
host memory. Apparent file lengths are not a measurement of allocated disk
clusters or filesystem overhead.

General developer worktrees that are unrelated to a study evidence workflow are
workspace overhead, rather than study evidence. This does not exclude a copy
merely because it happens to be in a checkout: a study-created or retained
publication, staging or audit copy must be inventoried. One workspace snapshot
found 160 registered-checkout copies of the preflight logits totaling
514,631,680 apparent bytes; that observation is not an allocated-cluster
measurement or a complete inventory. Do not delete other teams' checkouts.
Actual free space on the destination volume must be checked again at each
claim. If reconciled study evidence cannot fit, stop and publish a reviewed,
versioned storage-only allowance before continuing; do not call a host-disk
allowance a GPU allocation.

The existing natural claim fixed these prospective phase shares:

| Phase | Unique tensor share | Evidence-file share |
| --- | ---: | ---: |
| Natural discovery | 128 MiB | 224 MiB |
| Natural held-out | 160 MiB | 224 MiB |
| Injection discovery | 48 MiB | 128 MiB |
| Injection held-out | 72 MiB | 192 MiB |

The full shares remain charged even when a phase fails or retains less than its
limit. Before the next phase, a trusted custodian must account for all previous
study evidence not covered by a phase share, including every orphan and failed
attempt. The fixed shares leave 104 MiB for prior unique tensor payloads and
256 MiB for prior evidence and overhead; the ledger reserves 4 MiB of the latter
for its database, transaction journal and bounded inventory metadata. The
natural attempt and its publication/staging copies belong in its existing
phase share, not again in prior usage. Additional copies created after an
inventory must remain within the relevant share and be reconciled by the
custodian; the ledger cannot discover undisclosed copies.

`looking_glass.agreement_phase_ledger.initialize_study_ledger` imports the
already-used natural discovery claim and binding with independently supplied
SHA-256 pins, checks the completed, reaped/drained owner receipt, and inventories
the supplied quiescent natural and prior evidence paths. The known preflight
payload must be present. Its `prior_unique_tensor_bytes` argument is a trusted
audited subtotal: duplicate preflight copies count once logically, and all
other previously retained unique tensor payloads must be included. The caller
must supply every study-evidence path; this cooperative component cannot prove
that its list is complete. Its conservative imported-natural check counts all
natural evidence-file bytes against the natural logical share as an upper bound,
which may reject an otherwise valid larger physical copy. File contents and
sizes are checked again before any later claim; a changed inventory requires new
review. No natural model forward is run by initialization.

Each launcher calls `claim_phase(ledger_path, phase, output_path=attempt_dir)`
once at its execution boundary. The output must not already exist, its parent
must exist, and at least 1 GiB must be free on that volume. A SQLite transaction
with `synchronous=FULL` rechecks free space after the inventory scan and inserts
a unique phase row. This checks a moment in time; it does not reserve disk
space against other processes. An already-issued phase is
never refunded, including after cancellation, failure or an ambiguous return.
The imported natural phase cannot be claimed again. The returned `PhaseGrant`
contains the exact output path and phase limits. It is an ordinary copyable
value, not an execution lock: the trusted launcher must exclusively create that
same output once, never replay the grant, and enforce the limits across all
files and tensor snapshots, including its own start/terminal markers and
incomplete evidence. Concurrent cooperating callers must use the same ledger
file on a local filesystem. Copying, replacing, deleting or bypassing the file
can bypass coordination and is outside this mechanism's guarantees. SQLite
durability depends on the filesystem and hardware honoring synchronous writes.
An interrupted initial creation leaves the file reserved and requires explicit
reconciliation, not automatic deletion or reinitialization.

A storage grant does not prove a model was executed, evidence is complete, a
causal hypothesis is correct, or a held-out phase is scientifically authorized.
Prediction freeze, unique case and donor identities, model/device ownership,
per-forward limits, independent numerical review, and phase-specific publication
rules remain separate prerequisites.
