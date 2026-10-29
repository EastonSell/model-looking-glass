# Synthetic v2 one-shot custodian draft

This module is a **development-stage orchestration component**, not a command
to run the study. The separately reviewed discovery-evidence binding, production
executor, scorer and source-verified launcher are required before it can be used
for a real attempt. It fails before held-out work if the binding helper is not
available. Its tests use fabricated records, not new v2 cases or model work.

The caller supplies the prepared snapshot input and protocol identities. The
custodian calls the checked preparation and pre-manifest builders, creates new
private and public evidence directories, writes and rereads the public
pre-manifest and private material, and then executes discovery. It reads back
every private cell chunk, checks the discovery freeze against those chunks,
builds the public freeze and concrete private predictions, and verifies the
salted commitment and every prediction group before the first held-out model
callback. Both public freeze and commitment and the private prediction record
are written and reread first. One model per seed and phase is passed to the
executor across that seed's nine cells. Each held-out condition refers to three
consecutive committed predictions as
`sha256:<prediction-record-sha256>:<first-row-index>:3`.
The binder and freeze builder receive distinct parsed copies of one bounded
canonical selection snapshot; the custodian rejects mutation of either copy.

The executor is trusted to use each supplied seed, expected model-state hash,
private assignment and case exactly as specified; check the model/input pair,
private target integrity, all capture and intervention metadata, and all
required controls; return only sanitized discovery selections; and clean up
hooks and models before returning from `end_model` and `close`. The callbacks
are `begin_model(phase, seed, expected_model_state_sha256, budget)`,
`run_discovery(assignment, cases, cell, budget)`,
`run_heldout(assignment, cases, prediction_groups, cell, budget)`, `end_model()`
and `close()`. A prediction group is a `reference` plus its three saved site
rows. All callbacks execute on one custodian thread; the caller must guarantee
exclusive, quiescent model ownership. The budget's `reserve()` is called before
each preparation and `root()` immediately before the corresponding root call.
A success requires precisely 57,408 submissions and root forwards, and both
counters are limited to 60,000. This is a cooperative counter: it cannot detect
uncooperative calls that bypass it. The same budget instance spans every model,
cell and phase. Actual trace run IDs must use an attempt-wide namespace; a
clean or repeated-clean ID may be reused only for the same retained forward,
model, input and phase. Trace source IDs bind to their saved `run_id` and site,
and clean/perturbed base IDs use the matching `<run_id>/logits`. The custodian
rejects any trace run ID or condition ID occurring in both hash-checked discovery
and held-out chunks; the discovery and scoring validators separately check
reuse within each phase.

After successful cleanup, `close()` must return exactly
`{"input_model_pairing_verified": true, "private_plan_metadata_verified": true}`
as the executor's own attestation that its checks completed. Missing, false or
malformed receipts prevent scoring. This receipt is not an independent proof
of the callback's behavior.

After model cleanup, the trusted scorer consumes the persisted, hash-checked
held-out chunks in order. Immediately beforehand a required trusted exposure
classifier examines the attempt's exposure records and returns canonical
`synthetic-v2-exposure-v1` evidence with a classification of `unexposed`,
`unblinded`, or `unknown`, and its bounded basis. The private evidence retains
that record; the successful public reveal includes only its classification and
SHA-256, while the scorer receives the classification. The digest binds the
public label to the private record for a cooperating auditor; it is not an
authentication or anonymization mechanism, and predictable private text may be
guessable from the digest. Missing or invalid classification prevents scoring, and
unblinded or unknown evidence must not be presented as confirmation. The
classification is a cooperative attestation, not proof of secrecy.

Protocol records use the protocol's ASCII-escaped canonical JSON. Cell chunks
and private failure envelopes use the registry's UTF-8 canonical JSON; their
validators must not conflate these distinct encodings.

The scorer also receives explicit caller assertions about the
source/case/phase, input-model and private-plan checks; these are not independent
proof. The custodian requires the scorer to consume all chunks, writes results
only after checking the summary's exact schema, exposure label, interpretation,
source-record hashes and ordered model/cell/condition inventory against the
persisted chunks and cases. It then writes the reveal and verifies separate private-evidence and public-study
terminal outcomes. `succeeded` records completion of this cooperative path, not
proof that the callbacks ran as claimed. A source-verified launcher, actual
executor and scorer must bind the exact executed sources and model identities;
arbitrary callbacks must not be treated as scientific evidence.

Each attempt is single-use. The first unexpected exception, failed validation,
storage uncertainty or cancellation stops further forwards. A completed prefix
of a failed cell is published as incomplete when the private writer is known
safe. The current executor does not retain intermediate fragments from an
unfinished condition; a bounded partial record explicitly marks those facts
unavailable rather than implying they were saved. The full private failure
records both a primary and cleanup error if they fit. Public failure records
contain an opaque cell ID and generic stage only.
No exception is silently truncated. If failure serialization or publication is
uncertain, the attempt remains incomplete. Public publication exceptions and
retained `.pending` files stop further public writes; the underlying registry's
automatic pending cleanup is not used as a retry. If the public terminal call
raises after the exact intended success outcome and artifact hashes are visible,
`CommittedSuccessError` reports visible success while acknowledgement or
directory durability remains uncertain; an interrupt remains primary. Reading
back bytes after a failed POSIX directory sync does not prove power-loss
durability, and Windows has no portable directory sync here. A crash never
resumes the same attempt. The caller must inspect and supervise an uncertain
outcome.

The private writer reserves the declared payload and staging space inside the
1 GiB host-disk envelope, including a 128 MiB allowance for public evidence.
Its separate registry limit is 512 MiB and 128 artifact names. Public caps and
the largest staging alias fit that allowance. A 2 GiB free-space snapshot is
required on the destination before private initialization, but later disk
exhaustion remains possible. These are host-disk limits, not VRAM limits. The
callback owner must separately bound live Python, serialization and tensor
memory: the disk allowance does not measure RAM or files outside its declared
reserve. Publication flushes files and syncs directories on supported POSIX
systems; Windows has no portable directory sync in this implementation. These
records do not guarantee power-loss durability on all filesystems.

The public/private separation and local timestamps are procedural, not an
authentication or secrecy boundary against the same OS account. A commitment
alone does not prove when evaluation occurred, that random secrets were drawn
independently, or that the asserted source and model actually ran. Cancellation
at arbitrary interpreter bytecodes can leave locks or ownership uncertain;
supervision must quarantine or replace that process rather than retry it.
