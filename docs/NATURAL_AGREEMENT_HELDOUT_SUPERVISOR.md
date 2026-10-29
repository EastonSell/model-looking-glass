# Natural held-out CPU supervisor

`looking_glass.natural_heldout_supervisor.supervise_heldout` is a private,
single-attempt Windows process supervisor. The custodian supplies the command,
environment, an absolute fresh output directory and a structural audit
callback. The cooperating child waits in the frozen
`natural_agreement_controller.await_start()` before doing model work. The
supervisor assigns it to the kill-on-close job before sending the start token.

The custodian must establish the published prediction and aggregate freeze,
current exposure inventory, study-wide one-shot claim and resource allowance,
and source/checkpoint/runtime leases independently. This supervisor does not
allocate cases, draw a nonce, validate those prerequisites or authorize a
retry. Its `retry_safe` field is always false for this irreversible phase.

The child writes its evidence under `output/attempt`. Once the child has exited
zero, been reaped, the job has drained and closed, and both bounded logs have
drained, the supervisor calls `audit_result(attempt_dir, exit_code, reaped)`.
The callback returns a bounded JSON object with status `evidence_complete` or
`incomplete`; a complete result must include
`inventory.retained_tensor_bytes`. The supervisor independently counts the
quiescent retained tensor files, including partial files, and requires that
count to match. Evidence completion is structural; a separate raw numerical
and landing audit is still required.

Each log retains at most 1 MiB. A timeout, overflow, nonzero exit, interrupted
run, callback failure or accounting mismatch remains incomplete. If process
cleanup is unverified, no callback or tensor-file read is attempted and the
record says protection is unverified; owning process/job objects remain rooted
in a process-local quarantine, and later owner starts in that process are
refused. Streams with an unconfirmed pump remain open and their hashes are not
read. There is no automatic retry of ambiguous handle closes. Stop and resolve
process custody before any further study action. If writing the supervisor
record itself fails, the absence of a terminal is incomplete and custody stays
rooted rather than being silently released.
Cancellation is recorded separately even when uncertain cleanup takes status
precedence; the cancellation flag denotes `KeyboardInterrupt`, while
`SystemExit` is retained as an incomplete supervisor or audit failure.
Job accounting is a cooperative local process check,
not a guarantee against escaped or separately launched native processes.

The exclusive `supervisor.json`, logs, command and any partial evidence are
private and may contain local paths or held-out information. Preserve the
original files; publish only separately reviewed projections and hashes.
