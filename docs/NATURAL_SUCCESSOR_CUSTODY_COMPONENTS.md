# Natural successor custody components

These components support the separately frozen [natural successor protocol](NATURAL_AGREEMENT_SUCCESSOR_V1.md).
They do not authorize a private comparison, split draw, model run or claim that
the original failed held-out attempt succeeded. Its claim and sealed split stay
unchanged.

`natural_successor_claim.py` provides exclusive, flushed markers for a fixed
pool query and a separate phase. Its file flush is not proof of directory-entry
durability after an OS crash or power failure. Complete-looking markers after
an unobserved process exit remain ambiguous and require reconciliation.

`natural_successor_accounting.py` counts retained apparent file lengths against
the protocol's separate successor allowances. A flat content-addressed tensor
name is charged as a tensor only if its bytes match the name; unknown or nested
files count as side evidence. The caller must prevent concurrent writers,
reconcile old evidence and staging/orphans separately, and include every
retained rehearsal, observer and log file in the published limits. Free-space
checks are observations rather than reservations.

`natural_successor_execution.py` reconstructs the committed successor cases
and invokes the original four-family runner. It verifies supplied identities
and bidirectional known-row exclusions, but a receipt supplied to it is not
proof that a comparator ran, that a phase was validly claimed, or that the
inventory or peer state is current. It has no self-reservation exemption and
fails closed if its own pool appears in the supplied inventory.

`natural_successor_query.py` composes reviewed source leases, a start-gated
Windows Job supervisor and a one-use query marker around a fixed comparator.
The marker remains consumed after the query begins, including on collision or
uncertain process cleanup. A separate, source-bound outer caller must enforce
the protocol's stop without retry even for failures that precede this marker,
retain the caller process and observe its successful exit and reaping. A clear
file alone does not establish that observation. Quarantined process or lease
cleanup requires external reconciliation; forced termination and same-account
tampering remain outside cooperative custody.

Before using any component with the protected split, independently publish
and check the exact operation, fixed pool/input mapping, source dependencies,
root and file identities, ACLs, current reviewed inventory and comparison
process. The trusted custodian must run the complete no-model rehearsal of the
actual production bootstrap, selected site, interpreter and parent/child
lineage, and retain an independently checked startup-passed and clean-exit
record. The actual successor owner, rehearsal verifier, query outer observer,
cross-track first-forward gate, phase preparation and independent saved-result
auditor are separate obligations. No private pool query or successor nonce is
created by publishing these components.

The separately composed outer observer treats any nonzero query-parent exit as
an incomplete observation. A comparator collision makes that parent exit
nonzero, so this composition retains custody and stops rather than returning an
independently authenticated collision result. It must never infer clearance
from a missing or incomplete observation. The later phase caller must lease
the invocation marker and all retained handoff files through verification and
the phase claim, and independently establish the outer caller's own exit and
reaping.
