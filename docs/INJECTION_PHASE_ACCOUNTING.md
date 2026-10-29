# Injection phase-side accounting

This read-only component checks apparent regular-file bytes for the one-use
injection held-out phase. It scans the protected phase parent and the separately
retained rehearsal root twice, rejects symlinks, reparse points, hardlinks,
nonregular files, changed identities and oversized inventories, and requires the
same metadata in both scans. It counts all files under the fixed pilot sibling
against 16 MiB, the rehearsal root against 4 MiB, and all other protected-parent
files—including preclaim intent, receipts, launch markers, failures and the
preparation phase—against the remaining 172 MiB. Their sum must fit the existing
192 MiB phase share. The caller passes the independently verified replay tensor
subtotal, which must fit the 8 MiB logical reservation and leaves 64 MiB for the
held-out execution.

Before phase creation the phase must be absent; after the allocation announcement
and immediately before the nonce it must contain only the bounded `claim.json`.
The exclusive preparation-start marker must be present at both checks. Pilot and
rehearsal evidence must exist, and each protected root's caller-supplied device
and inode must match. A failed or partial file is still counted; its existence
does not mean the associated operation succeeded. An incomplete replay or
uncertain cleanup is terminal even when the files happen to fit.

The source-bound owner must independently establish the fixed paths, root ACLs
and identities, source/ledger bindings, replay outcome, current free space,
logical tensor subtotal, and any retained evidence outside these named roots.
Those duties cannot be inferred from a successful size check. The scan measures
apparent file lengths, not allocated filesystem blocks, and the same-account
cooperative custody model does not prevent an adversary from modifying a file
between observations. No contents, private site or token rows are read here.
