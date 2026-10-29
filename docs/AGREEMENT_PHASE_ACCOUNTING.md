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
The first [physical-only amendment](../experiments/agreement_physical_budget_v1.json)
raised the study evidence allowance to 2 GiB after natural discovery. Its
conservative 2026-10-29 candidate inventory found 321,350,312 apparent bytes
of natural evidence, including 24 published-archive checkout copies, and
224,697,899 apparent prior bytes. The candidate requires completeness review;
its recorded sizes do not count future copies. At 04:43:12 UTC the destination
volume reported 138,885,816,320 bytes free. The amendment changes storage
accounting only: the original claim and binding, logical tensor caps, cases,
analysis, predictions and results remain unchanged. A
[second physical-only amendment](../experiments/agreement_physical_budget_v2.json)
raises the allowance to 4 GiB and the natural-discovery share to 1 GiB. A later
candidate found 461,958,584 apparent natural bytes (31 archive copies),
254,239,892 prior bytes and 1,314,705 encoded inventory bytes. A shared Git
pack containing study blobs is charged once. This later candidate is still
incomplete and may change. The volume reported 136,344,756,224 bytes free at
04:57:39 UTC on 2026-10-29; this is not a reservation or a current free-space
assertion. The second amendment binds the first, the original claim and binding.
A [third physical-only amendment](../experiments/agreement_physical_budget_v3.json)
increases the study evidence allowance to 8 GiB and the natural-discovery
share to 4 GiB. A later incomplete candidate contained 63 archive copies,
860,777,090 natural and 821,583,110 prior apparent bytes, with 1,928 natural
and 6,281 prior files. At 05:23:54 UTC on 2026-10-29, the volume reported
128,697,253,888 bytes free. This is a historical observation, not a current
free-space assertion or evidence that the candidate was complete. The third
amendment binds the earlier chain and changes no logical limit, cases,
analysis, predictions or results.

The subsequent [evidence-scope clarification](../experiments/agreement_evidence_scope_v1.json)
applies prospectively after the natural and injection discovery phases. It
retains the 8 GiB apparent study-evidence allowance, phase shares and 12 GiB
fresh-free requirement. Study-created or study-retained evidence files,
including audit scripts and results, dedicated source snapshots retained as
evidence, publication copies, staging, failures and orphans, remain in scope
wherever they are kept. Developer source/test checkout files and shared Git
object stores for general workspace operations are separately reported
overhead, including when a study owner works there. Specific evidence copies
created or retained for the study remain in scope wherever they are kept,
including in a checkout or a dedicated publication archive.
This is not a limit on total workspace or machine usage. Free space on each
relevant volume must still be checked immediately before use.

The public [initial-inventory gap record](../results/agreement-physical-accounting-v1-20261029/initial-inventory-gap.json)
remains part of the evidence. Four public-result roots were discovered after
the injection claim, with unknown creation times relative to that claim. The
initial inventory was incomplete and this clarification does not cure it.
The scope amendment binds the exact pre-reconciliation `meta.data` and
`meta.inventory` strings with length prefixes, sorted claim rows with a separate
digest, and ordered reconciliation/retired-evidence rows with a history digest.
It does not hash the entire database file or SQLite schema. The old ledger did not itself store the initial manifest digest, so the
association between that digest and the saved state is an independent external
attestation. This update does not authorize replacing or refunding either
discovery claim or redrawing a private site; prior draw chronology is an
operator report, not established by the ledger.
The scope amendment's statement that prior commitment bytes remain unchanged
is an operation constraint: reconciliation does not open or independently
verify private commitments. The custodian verifies their integrity separately.

The observed candidate inventory and its generator are retained locally as
`work/agreement-ledger-inventory-candidate.json` and
`work/agreement_ledger_inventory_candidate.py`; the later candidate is
`work/agreement-ledger-inventory-next.json`, generated by
`work/agreement_ledger_inventory_next.py`; the third candidate is
`work/agreement-ledger-inventory-observed-v3.json`, with generator snapshot
`work/agreement_ledger_inventory_observed_v3.py`. The hashes are recorded in the
respective amendments. Absolute local paths are not published, so the public
aggregate alone does not reproduce the private path list. A custodian must
review the current inventory independently before issuing any phase grant.

Actual free space on the destination volume must be checked again at each
claim. If reconciled study evidence cannot fit, stop and publish a reviewed,
versioned storage-only allowance before continuing; do not call a host-disk
allowance a GPU allocation.

The original natural claim and the prospective amendment specify these shares:

| Phase | Unique tensor share | Original file share | First | Second | Current |
| --- | ---: | ---: | ---: | ---: | ---: |
| Natural discovery | 128 MiB | 224 MiB | 512 MiB | 1,024 MiB | 4,096 MiB |
| Natural held-out | 160 MiB | 224 MiB | 224 MiB | 224 MiB | 224 MiB |
| Injection discovery | 48 MiB | 128 MiB | 128 MiB | 128 MiB | 128 MiB |
| Injection held-out | 72 MiB | 192 MiB | 192 MiB | 192 MiB | 192 MiB |

The full shares remain charged even when a phase fails or retains less than its
limit. Before the next phase, a trusted custodian must account for all previous
study evidence not covered by a phase share, including every orphan and failed
attempt. The current shares leave 104 MiB for prior unique tensor payloads and
3,552 MiB for prior evidence and overhead. The ledger deducts a 128 MiB
allowance from the latter for its database, rollback journal and associated
files; serialized inventory and metadata together are bounded to 16 MiB, and
each natural/prior traversal is separately bounded to 32,768 paths.
A 16 MiB synthetic record and rollback-journal rewrite were tested against the
allowance. SQLite and filesystem allocation are not hard-capped by the ledger:
the custodian must reconcile actual files and fail if they exceed the allowance.
The
natural attempt and its publication/staging copies belong in its existing
phase share, not again in prior usage. Additional copies created after an
inventory must remain within the relevant share and be reconciled by the
custodian. Immediately before a future claim the custodian must freshly review
study-owned evidence, phase side roots and orphans, and coordinate any interval
needed to keep the reviewed evidence quiescent. The ledger rechecks only its
supplied roots and cannot discover an undisclosed copy elsewhere. A changed
study-owned inventory requires a reviewed update; incidental repacking of an
unrelated workspace Git object store is not a research gate.

`reconcile_study_ledger` atomically updates an existing ledger under its SQLite
write lock. Its pinned manifest and prospective scope amendment bind the exact
previous metadata/inventory strings, existing claim rows, published gap, earlier budget
amendment and reviewed current roots. The custodian independently reviews the
inventory and its logical subtotals; a digest does not attest completeness.
The transaction preserves all claim rows and original budget/evidence pins.
Every historical path, size and content digest absent or replaced in the new
inventory retains its previous physical charge; the exact same path and bytes,
if present again, are counted once. Prior snapshots and compact path/content
digests are retained in bounded history. Each claim checks the current history
digest, and the next reconciliation requires an independently pinned digest.
This is conservative apparent-byte
accounting, not proof that missing files exist or a measurement of allocated
disk blocks. Database access is cooperative; these digests do not authenticate
arbitrary out-of-band edits to its tables or hardware-level durability. Once reconciled, schema v6 makes an older v5 claimant fail closed
even if historical charges happen to be zero. The next execution owner must
pin the revised claimant source. A committed reconciliation with an ambiguous
return is inspected rather than replayed; never reset or replace the ledger.
Completed phase outputs and their private side roots stay charged to their
already consumed shares and are independently checked by their custodians,
not folded into the natural/prior inventory or opened by this transaction.
The ancestry checks rely on cooperatively reviewed roots: a hard link outside a
private parent can still alias a private file. The custodian must exclude such
aliases; this is not a hostile-filesystem confidentiality boundary.

`looking_glass.agreement_phase_ledger.initialize_study_ledger` imports the
already-used natural discovery claim and binding with independently supplied
SHA-256 pins. It validates their original 1 GiB physical budget, and requires
separately pinned published first, second and third amendments bound to those exact
bytes and to each other before applying the current limits. Independently pinned published audit and launcher-terminal
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
evidence-file copies are counted separately against its current 4 GiB physical
share. File contents and sizes are checked again before any later claim; a
changed inventory requires new review. No natural model forward is run by
initialization. This revised record uses ledger schema v5; an incomplete or
older ledger is not automatically upgraded or replaced.

Each launcher calls `claim_phase(ledger_path, phase, output_path=attempt_dir)`
once at its execution boundary. The output must not already exist, its parent
must exist, and at least 12 GiB must be free on that volume. This threshold
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
