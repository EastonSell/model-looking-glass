# Independent injection held-out saved-evidence audit

`tools/audit_injection_heldout.py` is a standard-library-only, private
verification component. It reads a quiescent journal only after the custodian
has independently established the frozen identities and process cleanup. It
does not prepare inputs, load a model, draw a site or split, or authorize a
forward. All inputs, exception details, and the complete audit report remain
private until the protocol permits reveal.

The `audit_saved` entry point independently verifies the two salted
commitments and the frozen prediction binding; it checks all 128 ordered
frames, checksums, independently retained receipts, and public capture
coordinates. It reconstructs the fixed sign vector, exact selected-token
addition and realized delta, patch and null delivery, untouched token bytes,
same-site order and adjacent witnesses, donors, upstream conditions, and
clean/repeat/null/self/upstream/downstream controls. It recomputes every
case's margins, first difference, vector norms, frozen-prediction and zero
errors, family groupings, reverse effects, and full-vocabulary restoration,
and requires the complete saved evaluation to agree. The 72 MiB tensor
check conservatively counts occurrences; it is not an RSS bound and may
overcount duplicate payloads.

The `expected` digests passed to `audit_saved` must be supplied independently
from the published freeze and commitments and the observed post-score
handoff. The `externally_quiescent=True` argument is a caller assertion, not
proof of quiescence. The component does not verify actual model execution,
the source or runtime loaded by a child, tokenizer behavior, the sampling
algorithm, current known-exposure inventory, split disjointness, commitment
publication order, durable storage, or truthful process supervision. The
discovery prediction is digest-bound here; its public raw discovery archive
was audited separately. The report explicitly records the prospective
orchestration departure from the original same-full-source rule.

`verify_score_handoff` additionally checks exact binding, evaluation,
supervisor, logs and post-score receipt against independently supplied
digests and command identity. It compares the supervisor's complete structural
audit with independently counted frame, tensor-occurrence, and owner side-file
bytes, and binds the owner terminal's model-state digest to a caller-supplied
expected value. It checks the recorded child exit, Job drain and
cleanup fields, but these records cannot prove that the scoring supervisor
itself returned successfully. An independently observed, source-bound outer
process completion is still required before accepting or publishing an audit.
The operational wrapper and final qualified report must also bind the
discovery and actual held-out runtime hashes separately, together with the
published prospective runtime-deviation and fixed public-case compatibility
check identities. The discovery runtime recorded in the frozen journal
binding is not evidence that the held-out runtime is identical.
The future owner must bind and verify that observation, publication and the
shared gate, then call `write_private_receipt` once to create the bounded
`independent-audit.json` exclusively. A failed or partial write is not an
acknowledgment and must be retained and reconciled. The owner charges the
receipt and other side files to its reserved phase budget.

There is deliberately no operational command-line entry point until the
outer-observer receipt and owner integration are reviewed. A caller must not
promote the numerical component's return value alone to a completed study.
Tests use invented commitments and an untrained CPU toy model; the integration
fixture reuses public discovery token rows as synthetic data, never an actual
held-out split.
