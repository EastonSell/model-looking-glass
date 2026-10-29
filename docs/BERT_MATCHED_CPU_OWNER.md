# Matched-dose CPU owner

`scripts/bert_matched_cpu.py` binds a clean committed checkout, the exact public
matched manifest and protocol, the prior exposure digest, and all five pinned
checkpoint files to the cooperative 64-row reservation. It starts a direct,
CPU-masked interpreter under the reviewed CPU supervisor and Windows job, with
source files leased until the owner is reaped and its closed journal is audited.
The owner verifies tokenization, runtime, calling-thread numeric probes, model
state and hook cleanup. Its journal contains only ordered runner events and is
capped at 16 MiB on disk; that cap does not bound peak memory.

Use separate ledger and custody directories for `--rehearsal`,
`--checkpoint-preflight`, and `--execute`. The first loads no model dependency;
the second loads the verified checkpoint with its forward method forced to
raise. Both report zero forwards and do not produce a study result. The actual
run requires an explicit `--execute` and preserves partial journals on failure.
The source checkout and all source hashes must be published and reviewed before
running the public cases.

`controller-result.json` is provisional: it becomes a completed observation
only after the controller itself is observed exiting zero and no failure marker
exists. A publication or cleanup interruption can leave both a candidate and
failure record. Uncertain child cleanup retains the in-process ledger and source
leases while recovery is attempted; forced termination of the controller and
same-account changes are outside this cooperative guarantee. In particular,
the child does not inherit the source-file lease: controller death can release
that lease before the job has finished terminating the child. This run assumes
a trusted, quiescent checkout, and source hashes do not authenticate the bytes
that were already executing. An unresolved or
completed reservation currently blocks an automatic retry, so it cannot erase
the first-attempt exposure history. File hashes and process records are not
independent attestation of executed native code or earlier exposure.

Repeated ordinary console interrupts are deferred once cancellation or failure
is being recovered. Forced termination or arbitrary interpreter-level
interruption can still stop the controller before recovery runs; an uncertain
attempt remains incomplete, and a receipt without a separately observed clean
controller exit is not a completed run.
