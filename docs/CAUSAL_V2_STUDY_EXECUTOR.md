# Synthetic v2 model callback

`looking_glass.causal_v2_study_executor.StudyExecutor` is a private,
single-threaded model callback for the proposed synthetic v2 custodian. It
does not generate a split or injection schedule, authorize execution, prove
blindness, verify source provenance, or publish a premanifest. A reviewed
launcher and custodian must establish those prerequisites and keep private
arguments and evidence private. No actual v2 split or held-out experiment is
run by adding this component.

The callback accepts the same attempt-wide `ForwardBudget` for every model and
phase. It verifies each actual CPU model state against the supplied expected
SHA-256 and binds each retained clean trace, target capture and patch plan to
the model, input, numeric policy and budget. It snapshots bounded caller
records before execution and rejects changes to them. Trace identifiers use
the attempt-wide submission counter; plan source and base references use the
corresponding retained trace ID and its `/logits` suffix. Input digests use
the byte contract documented in `CAUSAL_V2_DEVELOPMENT_EXECUTOR.md`.
Construction requires the backend settings snapshot destined for the
pre-manifest. The callback compares it with the live settings at callback
boundaries and immediately before and after each root call. At those same
boundaries it requires the CPU-only device mask, no initialized CUDA runtime
and the float32 default dtype. These cooperative checks cannot detect a
transient change restored between observations.

For each model and phase, two clean/repeat forwards are reused for each of 16
inputs across nine cells. Each cell/input has one private target capture and
two active strengths of 16 forwards each. A successful complete schedule thus
uses `2*16 + 9*16*(1 + 2*16) = 4,784` root forwards per model/phase, or
57,408 across six models and two phases. The budget reserves each submission
before preparing its model input or hooks, then records a root call immediately
before calling the model. An asynchronous interruption between the counter
increment and callable entry can conservatively overcount an attempted root;
the failure record is not proof that every counted callable was entered. A
failed or cancelled attempt is incomplete; the
caller must preserve its counts and quarantine or discard the affected worker,
model and budget as appropriate. Successfully encoded conditions have already
been added to the cell evidence. If a condition fails during execution or
encoding, this callback does not retain its intermediate control fragments; the
custodian must record them as unavailable rather than imply they were saved.

Discovery returns a bounded, sanitized selector result without the private
injection location. Held-out prediction rows are checked for shape, order and
their declared reference, but the custodian must verify those rows and the
reference against the committed discovery freeze before invoking the callback.
The separately reviewed scorer and custodian own cross-phase trace identity,
global split and exclusion checks, prediction commitment, negative-control
deduplication, exposure classification, and durable evidence publication.
Successful `close()` returns the callback's cooperative pairing and plan-check
receipt only after all model/phase work and cleanup succeed; this is not an
independent verification of the caller or execution environment.

Caller-provided JSON inputs are structurally bounded before serialization, and
each sanitized selection has a 512 KiB logical limit. These limits do not cap
the process's total host memory. The executor assumes exclusive cooperative
ownership of its model and callback thread; neither these checks nor source
hashes are a security boundary against arbitrary in-process mutation. Evidence
byte limits, durable incomplete records, source verification and operating
system resource scheduling remain separate responsibilities.
