# Independent injection runtime replay audit

`tools/audit_injection_runtime_replay.py` checks the private saved evidence for
the one fixed, already public discovery case used in the prospective runtime
compatibility check. It loads no model and performs no forward. Its fixed
reference hashes identify the sixteen public case-0 observations at discovery
commit `6b0dd7e1c7e30288d4f563175d589540ee8480fd`; the request plan, discovery
source and discovery runtime remain fixed. The auditor verifies the private
site commitment without disclosing the site or nonce.

Using the separately sourced held-out auditor's integer arithmetic and witness
checks, it independently verifies the sixteen frames and acknowledgments, the
independently retained receipt chain, request and metadata identities, selected
and untouched coordinates, donor pairing and same-site ordering. It compares
every attention and MLP capture, the final block and all last-token logits to
the pinned references under the specified tolerance, and recomputes the private
terminal's per-role comparisons. It rejects unexpected paths and files, checks
bounded ordinary file identities, counts encoded pair bytes conservatively and
limits the examined attempt directory to 16 MiB. The numerical audit probes
nearest binary64 rounding at its start and end; a transient intervening change
is not ruled out.

The return value is a **private saved-evidence result**, not a public success or
proof of execution. The caller must independently verify the published source,
actual held-out runtime and runtime-deviation plan, the protected commitment,
the exact source-bound supervisor terminal, the outer observer's clean result,
and the separately observed successful exit of that observer. The caller must
also establish quiescence, publication order, the single-use ledger claim and
whole-phase resource reconciliation. The model-state hash and quiescence
parameter supplied here are caller-provided identities or assertions; this
component does not establish their truth. The private report's resource totals,
frame hashes, errors and per-role details must not be published before the
authorized reveal. Any permitted pre-reveal projection is limited to the
separately verified overall pass and fixed public identities.

There is no operational command-line entry point or public-pass writer until
the supervisor and observer-exit handoff are frozen and independently reviewed.
An unknown or unsuccessful exit, a failed audit, or incomplete evidence leaves
the replay incomplete. Passing this one public-case check does not establish
runtime equivalence or correctness on any held-out input. Tests use a fabricated
site, an untrained CPU toy model and the already public case-0 token IDs only.
