# Independent natural-agreement evidence audit

`tools/audit_natural_agreement_discovery.py` checks a completed public
development journal against the fixed preflight manifest and its saved tensor
payloads. It imports only Python's standard library and NumPy; it does not load
the model or import the study implementation. For each candidate it
reconstructs the present- and past-tense effects and selection from the stored
float32 bits. It checks the planned forward order, declared control results,
donor landing, untouched coordinates, full-vocabulary control outputs, and
hook metadata. The auditor rejects missing, contradictory, nonfinite or
misidentified evidence rather than treating a partial journal as a result.

Run it after the owner process exits and the controller verifies its frozen
source/runtime binding, start record, terminal status and exit/reaping evidence:

```text
python tools/audit_natural_agreement_discovery.py <attempt-directory> \
  --preflight-manifest results/distilgpt2-agreement-preflight-v1-20261028/frozen.json \
  --output <new-audit-record.json>
```

The output is canonical JSON and will not overwrite an existing file. Its
success record hashes the audited event log, start and terminal records, the
preflight manifest and this exact source file. The event log indexes retained
tensor bytes and their hashes. Preserve the journal and every referenced
payload together with the audit; a hash alone cannot reproduce the data.
`--synthetic` is for fabricated test journals only and its acknowledgment
explicitly uses a synthetic scope, so it cannot stand in for the public
pretrained development run.

This is an independent reconstruction of saved evidence, not proof that a
model actually executed, that its reported runtime or state was authentic, or
that source publication preceded execution. The controller and custodian
verify those separate bindings and retain their limits. Success concerns the
fixed public development cases; it does not establish a mechanism, held-out
performance or general behavior on other inputs.
