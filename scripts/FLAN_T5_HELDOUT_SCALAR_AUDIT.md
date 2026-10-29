# Held-out FLAN-T5 scalar consistency check

This separate, standard-library audit checks the two salted custody
commitments, retained float32 logits through binary64 log-sum-exp, rounded
matched-dose vectors and the three prespecified case, pair and aggregate
decisions. It checks source blobs and publication-observation consistency when
run with a pinned checkout and commit. Preserve its output and diagnostics in
the protected custody area until the protocol's reveal.

The `1e-3` per-token log-probability comparison is a diagnostic tolerance, not
an exact float32 replay. It is wider than the study's `1e-4` ordering tolerance,
so this check alone cannot exclude a small score discrepancy at a decision
boundary. A separate independently frozen audit must check the exact
float32 probabilities and same-forward intervention witnesses and untouched
positions. Neither audit independently proves model execution, local secrecy,
salt-generation timing, external publication chronology, prior-pretraining
independence or generalization.

Example usage after a closed attempt, with externally checked source, script
and terminal digests:

```
python -I -B scripts/flan_t5_heldout_scalar_audit.py \
  --run-dir PRIVATE_ATTEMPT --custody-root PRIVATE_CUSTODY \
  --terminal-sha256 TERMINAL_SHA256 --auditor-sha256 AUDITOR_SHA256 \
  --source-checkout FROZEN_CHECKOUT --source-commit SOURCE_COMMIT
```

The `--synthetic` option is only for fabricated fixtures. The author exercised
the auditor before any held-out model forward against the separate 858-forward
fabricated model/tokenizer fixture, including a semantic exclusion and a
strict-gate failure with a positive descriptive result. Its outputs and
fake source identities do not establish behavior of the pretrained model.

The first published scalar version stopped on an exact L2 comparison in the
closed real attempt: exponentiation and multiplication differed by one
binary64 ULP. This corrected version uses the fixed protocol's multiplication
and was written after that failure was inspected. Preserve the first incomplete
report and label this version as a post-inspection diagnostic; the separately
frozen full saved-byte auditor preceded the real run.
