# Known-injection executor-to-selector adapter

`looking_glass.pretrained_number_adapter` maps the ordered executor's public
observation types to the separate selector's value types. It preserves the
measured 13 boundary coordinates, requested patch/null coordinates, typed
donor identity and immutable payload byte objects. It drops the executor's
ownership token and does not accept or export private injection witnesses.
The selector performs the condition, coordinate, pairing, size and numeric
checks after conversion.

The adapter does not run a model or represent a completed experiment. It does
not verify the checkpoint, source, model state or private witnesses; establish
secrecy or process isolation; save the raw artifacts; create or publish a
commitment; attest publication ordering; draw a held-out sample; or authorize
discovery. A trusted custodian must do all of those duties, keep exceptions
and private evidence private, and pass the selector an attestation only after
the actual controls have been checked. The selector's boolean is an assertion,
not a proof.

The integration test exercises one complete 16-forward case on a synthetic
CPU toy model and combines it with seven independently fabricated selector
fixtures. It checks that all roles reach the selector, an incorrect donor is
rejected, and private evidence cannot be mapped as a public observation. Its
`True` private-control flag is a fabricated test assertion. No pretrained
model, actual site commitment, generated held-out case, GPU, or experiment
result is involved.
