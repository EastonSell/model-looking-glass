# Independent natural-agreement held-out audit

`tools/audit_natural_agreement_heldout.py` checks a completed saved journal
after the held-out record and nonce have been revealed. It does not load a
model or select a site from held-out effects. It binds the exact prediction
file to the externally pinned aggregate freeze, recomputes the held-out
commitment, and checks that the journal's cases and identity fields match the
revealed records. Its numerical and tensor-witness calculations are separate
from the model runner; the saved-journal checks build on the independent
discovery auditor's method.

The audit checks all 608 scheduled forwards, tensor hashes and shapes, hook
metadata, donor landing and untouched coordinates, null/self/repeat/delivery/
subject/final-clean controls, and all 16 present/past site-metric events. It
reconstructs each pair, four-cell family contrast, clean baseline and sign
classification from saved float32 bits, and includes the verified clean cases
and present/past metric events in its report. For each of eight sites it scores the
four held-out family effects against the already frozen prediction and zero,
using the protocol's `math.fsum` MAE and RMSE. The frozen selected, earliest,
random and magnitude sites are reported unchanged, including abstention and
duplicate baselines. Four families are descriptive units; no significance or
population-generalization claim follows.

The caller must independently verify the published freeze and commitment
pins, publication order, nonce allocation and sampling, exposure inventory,
tokenizer, source and runtime identity, faithful execution, child exit and
cleanup. Hash agreement or a `complete_candidate` terminal does not establish
those facts. The auditor checks the revealed bytes and their linkage but does
not independently reconstruct the nonce-based sampling algorithm or prove that
the model ran. The audit rejects missing or extra journal artifacts, including
pending terminal files, and requires unaliased regular files and ordinary
directories. A failed or partial journal cannot pass. Keep the journal,
prediction file, reveal and nonce private until the protocol permits disclosure.

For the prospective natural held-out phase, the auditor enforces the 160 MiB
unique-tensor share in [the phase accounting plan](AGREEMENT_PHASE_ACCOUNTING.md)
and a 48 MiB event-log cap in the planned operational binding. These are
phase-specific operational limits within the original study-wide limits, not
evidence that the caller accounted for publication copies or other phases. The
frozen CPU runner has no `device_rng` field; if a later journal supplies it,
the auditor requires consistent values in both contract and summary, but does
not treat those values as independent runtime attestation.

After reveal, use externally verified values for `FREEZE_SHA256` and
`HELDOUT_COMMITMENT`; the nonce file contains exactly 32 raw bytes:

```text
python tools/audit_natural_agreement_heldout.py JOURNAL --prediction PREDICTION.json --freeze FREEZE.json --reveal REVEAL.json --nonce-file NONCE.bin --freeze-sha256 FREEZE_SHA256 --commitment HELDOUT_COMMITMENT --output AUDIT.json
```

The output must be a new file outside the saved journal; without `--output`
the report is printed to standard output. `--synthetic` is for fabricated
toy journals only and labels the result accordingly. The implementation bounds
individual reads, the event log and retained tensor payload; caching and
NumPy temporaries can make process memory exceed the retained-payload total.
Input directories and their ancestors must remain controlled and unchanged
through the audit. The command's source and the resulting report hashes are
separate artifacts, not attestations of publication or execution.
