# Injection preparation handoff

`tools/injection_preparation_operation.py` connects an already consumed,
independently observed replay claim to the fixed injection split custodian. It
does not create a claim, start a process, publish an artifact or independently
attest the caller's observations. The source-bound caller must load the exact
reviewed custody, replay-owner, claim, guard and accounting modules and must
separately bind its own source, protected configuration and process lifecycle.

`ClaimHandoff` carries the exact protected replay configuration bytes, plan and
receipt digests, workspace and parent/ledger identities. The caller separately
anchors the v6 claim row in `ClaimAnchor` and the disjoint retained phase roots
in `PhaseRoots`. Before the preparation marker is written, the operation
checks the fixed discovery identity, current published inputs and inventory,
the observed claimant exit, replay evidence, claimant receipt/intent and exact
ledger row. The replay callback must establish the independently audited
result and its tensor count; a caller-supplied count is not evidence by itself.

The custodian creates the one-use marker and rechecks the claimant receipt,
configuration, exact ledger row and bounded side files before phase creation
and again after the shared allocation gate. The final check and one nonce write
run while the existing SQLite guard is held. Callbacks used at that boundary
must be local and bounded, and the owner must keep the protected database and
source paths stable. On a failed or uncertain handoff, the caller retains all
evidence and must not claim or draw again.

This is wiring for the reviewed cooperative protocol, not a runnable held-out
owner. A real caller still needs the published replay result and source binding,
private site and tokenizer checks, the current exposure observation, the
shared gate and any required cross-split comparison, independent process-exit
observation, and a rehearsal of the exact final production command.
