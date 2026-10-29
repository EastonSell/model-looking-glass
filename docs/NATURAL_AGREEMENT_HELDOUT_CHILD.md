# Natural held-out child bootstrap

`tools/natural_agreement_heldout_child.py` is the separately published
operational child for the natural held-out custodian. The scientific source
remains the frozen discovery checkout at `def07cbb168408a12c76ba30f196af45149e111d`;
the additional operational source is a prospective deviation documented by the
custodian. This child never draws a nonce or chooses a new family.

The supervisor passes a single absolute path to a private JSON configuration.
Its exact keys are `schema` (`mlg-natural-heldout-child-config-v1`),
`frozen_root`, `cache_dir`, `phase_dir`, `preparation_supervisor`, `public_files`
and `public_urls`. The file map contains absolute paths for `binding`,
`prediction`, `aggregate`, `inventory`, `v7_inventory`, `inventory_source`,
`operational_binding` and `deviation_bytes`. The URL map contains immutable
official URLs named `binding_url`, `prediction_url`, `aggregate_url`,
`inventory_source_url`, `operational_binding_url`, `wrapper_url`,
`deviation_url` and `commitment_url`. Do not include private case contents or
the nonce in this file or on the command line.

The parent must first verify and lease the operational sources and frozen
checkout, ensure the one-shot ledger grant and publication gates, and launch
the direct base interpreter with the discovery bootstrap's `-I -S -B`, fresh
absolute pycache prefix, explicit virtualenv site-packages, CPU mask and one
OMP thread. Both `MKL_CBWR` and `ATEN_CPU_CAPABILITY` must be absent. The
child checks this before importing Torch and records both as null. These are
known held-out settings; the discovery values were not recorded, so they do not
establish identical historical numerical runtime.

Before importing a project module, the child hashes the pinned controller and
executes only its frozen `await_start` gate. Once signaled, its source loader
hashes each frozen module before compiling those exact bytes, without pyc. It
checks the published operational binding and commitment, takes its own source
leases, validates the discovery binding and pinned tokenizer, and asks the
custodian to verify the completed, supervisor-observed preparation. Only then
does it write the exclusive owner-start marker and load the model. The loaded
model state and tokenizer are checked before the frozen held-out runner and
frozen predictions are used. The child observes the official current exposure
inventory immediately before the first forward and again after the run; an
unavailable or changed snapshot makes the attempt incomplete. Those observations
are not an atomic global lock, so other teams must still coordinate exposures.
The parent must retain its leases through verified
child cleanup; this cooperative process design is not protection against a
hostile same-account process or independently launched native code.

The child checks at least 3 GiB free at both cache and destination, records
pre/post tokenizer and checkpoint staging-directory observations, and leaves
any newly unresolved staging directory in place for reconciliation. Logs,
configuration, preparation receipt, stage observations and partial results are
private; disclose only reviewed projections. An exclusive private failure
traceback is capped at 128 KiB and may contain held-out text; it is included in
the custodian's reserved side-file budget. An owner exit and structural audit
still require the independent numerical and landing audit.

For a separate rehearsal, `--validate-only <config-path>` verifies published
public inputs, current inventory, source, runtime and tokenizer, then returns
`validated_no_model_forward` without reading a commitment, nonce or held-out
record, writing an owner-start marker, or loading a model. Use a fresh rehearsal
directory and no phase grant. A supervisor will correctly report no discovery or
held-out journal for that validation-only process.
