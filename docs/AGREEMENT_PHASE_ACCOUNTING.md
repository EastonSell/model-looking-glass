# Agreement-study phase accounting

This prospective clarification applies to the pretrained number/agreement
study after its existing natural discovery attempt. The frozen protocol's
512 MiB limit concerns **unique retained tensor payloads**, including the
3,216,448-byte public preflight logits. The original binding planned a
separate 1 GiB allowance for study evidence on host storage: retained originals, private and
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
The prospective [physical-only amendment](../experiments/agreement_physical_budget_v1.json)
raises the study evidence allowance to 2 GiB after natural discovery. Its
conservative 2026-10-29 candidate inventory found 321,350,312 apparent bytes
of natural evidence, including 24 published-archive checkout copies, and
224,697,899 apparent prior bytes. The candidate requires completeness review;
its recorded sizes do not count future copies. At 04:43:12 UTC the destination
volume reported 138,885,816,320 bytes free. The amendment changes storage
accounting only: the original claim and binding, logical tensor caps, cases,
analysis, predictions and results remain unchanged.

The observed candidate inventory and its generator are retained locally as
`work/agreement-ledger-inventory-candidate.json` and
`work/agreement_ledger_inventory_candidate.py`. The JSON's SHA-256 is recorded
in the amendment. Its absolute local paths are not published, so the public
aggregate alone does not reproduce the private path list. A custodian must
review the current inventory independently before issuing any phase grant.

Actual free space on the destination volume must be checked again at each
claim. If reconciled study evidence cannot fit, stop and publish a reviewed,
versioned storage-only allowance before continuing; do not call a host-disk
allowance a GPU allocation.

The original natural claim and the prospective amendment specify these shares:

| Phase | Unique tensor share | Original file share | Revised file share |
| --- | ---: | ---: | ---: |
| Natural discovery | 128 MiB | 224 MiB | 512 MiB |
| Natural held-out | 160 MiB | 224 MiB | 224 MiB |
| Injection discovery | 48 MiB | 128 MiB | 128 MiB |
| Injection held-out | 72 MiB | 192 MiB | 192 MiB |

The full shares remain charged even when a phase fails or retains less than its
limit. Before the next phase, a trusted custodian must account for all previous
study evidence not covered by a phase share, including every orphan and failed
attempt. The revised shares leave 104 MiB for prior unique tensor payloads and
992 MiB for prior evidence and overhead; the ledger reserves 4 MiB of the latter
for its database, transaction journal and bounded inventory metadata. The
natural attempt and its publication/staging copies belong in its existing
phase share, not again in prior usage. Additional copies created after an
inventory must remain within the relevant share and be reconciled by the
custodian; the ledger cannot discover undisclosed copies.

`looking_glass.agreement_phase_ledger.initialize_study_ledger` imports the
already-used natural discovery claim and binding with independently supplied
SHA-256 pins. It validates their original 1 GiB physical budget, and requires a
separately pinned published amendment bound to those exact bytes before applying
the revised limits. Independently pinned published audit and launcher-terminal
receipts bind the completed, reaped/drained owner and its journal and
content-addressed tensor inventory; their file hashes and tensor totals are
checked against the supplied quiescent natural evidence. The known preflight
payload must be present. Its `prior_unique_tensor_bytes` argument is a trusted
audited subtotal: duplicate preflight copies count once logically, and all
other previously retained unique tensor payloads must be included. The caller
must supply every study-evidence path; this cooperative component cannot prove
that its list is complete. The required `natural_unique_tensor_bytes` subtotal
must cover every distinct retained natural tensor, including orphan evidence
and uncompressed payloads inside archives; deduplicate only exact identical
snapshots. It must be at least the supervisor's reported retained-tensor bytes
and fit the 128 MiB logical share. That report is a lower bound, not proof that
the caller's complete inventory contains no additional tensors. Natural
evidence-file copies are counted separately against its revised 512 MiB physical
share. File contents and sizes are checked again before any later claim; a
changed inventory requires new review. No natural model forward is run by
initialization. This revised record uses ledger schema v3; an incomplete or
older ledger is not automatically upgraded or replaced.

Each launcher calls `claim_phase(ledger_path, phase, output_path=attempt_dir)`
once at its execution boundary. The output must not already exist, its parent
must exist, and at least 3 GiB must be free on that volume. This threshold
allows additional space for checkpoint staging, but does not cap that staging
or account for unrelated concurrent consumers. A SQLite transaction
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
