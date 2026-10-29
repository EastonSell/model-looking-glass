# Targeted-edit v2 evaluation evidence

This prospective, CPU-only reader recomputes evaluation metrics from saved
float32 logits using the same v2 scoring implementation as the producer.
It therefore checks saved data against the report; it is not a separate
implementation of the numerical formulas. It checks the artifact inventory, row and
condition ordering, external terminal and study-identity pins, restoration and
supervision records, and the fixed numerical decision. It cannot prove that
the recorded model ran, that weights were edited or restored, that a remote
fetch occurred, or that an attestating process was actually reaped. The parent
must derive a final study outcome only after those separate gates pass.
The reader probes its own calling thread's denormal behavior before and after
the metric calculation; these probes do not establish every library thread's mode.

The trusted owner obtains `SelectionSnapshot` directly from
`prepare_committed_selection`, checks the exact evaluation witness using the
source-bound witness validator, and passes its verified commit, digest and
reported observation time to `audit_evaluation`. These Python inputs are
constructible by callers; the auditor alone does not verify their provenance.
The output directory and all of its ancestors must remain quiescent and
controlled by the owner throughout validation and publication.

The directory contains `started.json`, child-written `owner.json`,
parent-assembled `metadata.json`, `selection.json`, `report.json`,
`observations.jsonl`, `condition_states.jsonl`, five fixed-shape `.f32`
matrices in protocol condition order, and a `terminal.json`. Every document
and journal row binds the version, phase, study run ID, manifest digest, source
commit, selection commit and terminal digest, and evaluation-witness commit
and digest. Rows bind each fixed case ID, byte range, SHA-256 and calling-thread
denormal checks. Condition rows bind the before/after model, weight and
supplemental buffer digests, applied weight digest, runtime settings, probes
and restoration attestations.

`metadata.json` binds the selected/random/opposite control attestations and
recorded CPU environment to the frozen development snapshot, and records the
fresh owner identity alongside the parent's matching zero-exit, clean,
non-timeout, reaped supervisor result. The prior 1% condition uses the v1
reference gradient's recorded applied-weight digest. The child may write
partial evidence, but only the parent may assemble a complete terminal after
reaping the child and verifying disposal. All cleanup fields, including
`model_disposed`, must be true. Failed, cancelled or partial attempts are
preserved separately and must never be promoted to a complete result.

`report.json` contains the independently recomputable numerical report; its
decision is provisional until source, custody, execution and cleanup gates
also pass. UTC ordering is a consistency check of recorded clocks, not a
trusted timestamp. The saved norms, state hashes, flags and disposal fields
remain attestations: logits do not independently establish their truth.
