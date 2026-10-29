# Read-only injection preclaim check

The injection held-out phase is claimed once before the fixed public runtime
replay. This component checks a later preparation's existing claim without
issuing another claim, drawing a nonce or opening protected site or split data.
It reads the v6 SQLite ledger in one read-only snapshot, compares the exact
injection phase output and 72 MiB logical / 192 MiB physical row, verifies both
original discovery claim rows, and recomputes the reviewed metadata and history
digests. It also compares the digest of all five injection claim fields against
the independently retained claim receipt, so changes to recorded free space or
any other field fail. A natural held-out claim can coexist only with valid
recorded free space and nonoverlapping output; its authority must be verified
separately if needed. It checks the protected
ledger file's device and inode before and after the read, rejects a changed path,
link or replacement, and bounds the metadata and history it processes.

The source-first owner must establish its `ClaimAnchor` independently from the
protected, reviewed preclaim configuration: canonical ledger and output paths,
database device/inode, reconciled state and history hashes, scope amendment, and
the original two-claim hash and exact injection claim-row hash. These
caller-supplied values do not authenticate
themselves. A later legitimate ledger reconciliation can change the state and
history: independently review and bind the new snapshot before using it; do not
reset the database or replace a consumed claim. The owner also verifies ancestor
identities and ACLs, available space, all side evidence and the replay result,
the one-use preparation launch record, exact executable sources, and clean
process exit. A successful row check is neither a process lease nor permission
to start or repeat a split draw. This is a cooperative same-account boundary,
not a defense against an adversary replacing files between operating-system calls.

The production preparation uses a read-only check when consuming its durable
handoff. After releasing the shared split gate, it enters `hold_existing_claim`,
which takes a short SQLite `BEGIN IMMEDIATE` write reservation, enables
query-only mode, verifies the same row/state on that transaction, and holds it
through the single nonce call and durable nonce write. The outer watchdog
bounds the local ledger and two-pass evidence checks in this critical section.
Cooperating v6 claims and reconciliations wait only for this boundary. The lock does not prevent out-of-band database
replacement; the protected owner must forbid it and check the pinned identity.
A failed or uncertain nonce or lock release remains consumed and incomplete.
An interruption between the OS entropy return and local assignment cannot be
recovered; it never permits a redraw.
The retained handoff and the outer owner's separate durable launch marker
prevent retries. All callbacks and saved receipts require their own source and
lifecycle review.
