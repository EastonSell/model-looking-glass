# Targeted-edit v2 remote-fetch attestation

This prospective record binds a reviewer's reported remote fetch to the exact
source, manifest and custody receipt before development, and to every
terminal-inventoried development artifact before evaluation. It does not
prove that a network fetch occurred, authenticate the reviewer, establish a
trusted timestamp, or attest model execution. The reviewer must independently
fetch the objects from `EastonSell/model-looking-glass` and compare both
the Git blob IDs and SHA-256 values before reporting them as fetched.

The manifest must include this module's source. Given the pinned manifest,
custody receipt and, for evaluation, completed development terminal,
`expected_fetch_items` computes the local comparison list. The reviewer
independently verifies every item against the remote at the named commit.
`make_witness` accepts the verified list, reviewer label and observed UTC,
then creates a canonical public receipt. The list contains no private cases.
The selection terminal and every file it inventories are included for the
evaluation witness; exact equality of the two digest fields must be checked
against independently fetched bytes, not inferred from the receipt.
An evaluation witness also binds and revalidates the development witness,
requires it to precede the selection commit, and checks its reported time
against the saved development start.

Commit the receipt separately after the artifacts it describes and before the
next phase's designated directory exists; the directory path is bound in the
receipt. An external SHA-256 pin and
`verify_committed_witness` check receipt bytes, local ancestry, source and
artifact identities, and component-wise case-insensitive Git tree lookups.
Chronology paths use unambiguous ASCII components; a non-ASCII name at a
checked tree level fails closed. The owner must use a quiescent checkout and
independently check resolved parents for links or reparse points and ensure
that neither the physical output path nor any alias is already present before
exclusive creation. Git cannot establish historical 8.3 or
other filesystem aliases, uncommitted artifacts, or whether a directory was
created and removed between commits. When the phase
has a recorded start time, supply it to reject a self-reported observation
that is later. A consistent local clock or commit order does not prove actual
forward timing. Before evaluation, the owner must separately run the complete
selection gate as well as this witness check. It must also verify the runtime,
checkpoint, fresh exposure/custody state and any remaining study gates.
