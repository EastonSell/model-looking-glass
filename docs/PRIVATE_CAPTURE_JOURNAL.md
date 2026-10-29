# Private CPU capture journal prototype

`PrivateCaptureJournal` and `JournaledActivationPlan` are an opt-in, trusted,
CPU-only prototype for retaining completed capture steps when a later part of a
forward or plan fails. A journal record is private diagnostic evidence, not a
successful plan, a completed forward, or a causal explanation. The existing
`ActivationPlan.captures` and `metadata` properties remain unavailable after a
failed plan. This prototype is not wired into the research runners, GPU broker,
worker service, IPC, or device-memory admission.

The caller must execute a trusted CPU-only model and inputs in a CPU-only
process. The constructor checks the model's registered parameters and buffers
and explicit intervention sources, and the journal rejects non-CPU selected
captures. These checks cannot prevent a stateless or custom model from using a
GPU internally, and the inherited plan may inspect a selected activation before
the journal checks it. They are not a sandbox or an enforceable resource
boundary. Do not pass protected study inputs through this prototype without a
separately reviewed custody, device, and exposure protocol.

## Scope and privacy

One exclusively created file belongs to one journal and one plan. A completed
capture is copied and appended immediately after its capture step. A subsequent
step, model exception, quota failure, or write failure invalidates the plan; an
earlier complete frame can still be inspected privately. Conversely, absence of
a frame does not establish that its model operation did not occur: a failure can
intervene between the operation and its journal write. A clean file tail does
not establish that the plan or forward completed. The raw file and exception
tracebacks can contain activations, model inputs, source values, and hidden
intervention locations. Keep them in a trusted directory with suitable
filesystem permissions, exclude them from investigator-facing output, and do
not log exception locals. Exclusive creation prevents accidental overwrite; it
does not protect against a malicious parent directory, symlinks, or another
process with access.

Model hook ownership still uses the in-process `ActivationPlan` lease. It does
not exclude unrelated processes, arbitrary external hooks, direct uncooperative
model calls, or shared mutable storage. The caller must serialize model,
journal, and lifecycle operations and establish that no such activity can run.
The journal's checks reject some races; they do not themselves prove
quiescence. CPU tensors are copied synchronously from contiguous storage without
NumPy; the caller must ensure that external code does not invalidate the
storage during the copy.

## Frame and byte limits

The file begins with a versioned magic value and a checksummed identity header.
Each following frame holds a fixed-size length prefix, canonical JSON metadata,
raw tensor bytes in the header's native byte order, a SHA-256 digest, and a
footer. A full frame is reserved before its payload is copied. The writer flushes
and calls `fsync` after each frame. An exception, including a short or stalled
write, marks the journal failed and fails the plan. A full-looking footer can
still be present if `fsync` or its acknowledgment failed. Checksums and footers
show only that the bytes are structurally complete and match each other; they
are neither authentication nor proof of stable-media durability or clean
process exit. A supervisor must retain an independently observed completion or
failure acknowledgment for any stronger claim. Power-loss and filesystem
behavior remain outside this prototype's guarantees.

The required positive limits have distinct scopes:

* `max_journal_bytes` covers the magic, identity header, all frame prefixes,
  metadata, payloads, hashes, and footers, including reserved attempted frames.
  `max_records` bounds capture frames. Failed reservations do not authorize
  further writes.
* `max_metadata_bytes` bounds each encoded JSON header or event. Encoding stops
  on reaching that bound; a single escaped JSON fragment and the plan's own
  metadata may require additional temporary memory.
* `max_workspace_bytes` checks twice the selected tensor byte count before the
  contiguous CPU copy and immutable payload copy. It is a logical allowance
  for these copies, not a process RSS limit. The base plan has already cloned
  the selected capture before this journal allowance is checked. Plan-owned captures, model inputs,
  outputs, source tensors, metadata, allocator overhead, filesystem cache,
  and tensors retained by exception frames are outside it. The caller must
  budget those separately and retain any relevant claims until all aliases and
  tracebacks are released.
* The reader requires its own file, record, metadata, and
  `max_read_payload_bytes` bounds. The last limit covers retained returned
  payload bytes; parsing and hashing have extra temporary overhead. The caller
  owns and must budget the returned immutable bytes and their aliases.

The journal stores caller-supplied `case_id`, `input_sha256`, `model_sha256`,
and `plan_sha256` labels. It does not compute or attest the input, model, or
plan represented by those labels. `read_capture_journal` requires the trusted
caller to provide the exact expected identity and `externally_quiescent=True`.
It checks canonical typed metadata, frame digests and footer, sequence numbers,
unique capture names, declared dtype/shape/selector consistency, and configured
limits. It reports the verified prefix with `tail` set to `clean`, `incomplete`,
`invalid`, or `capacity`; `durability` is always
`not-established-by-file`. An invalid or incomplete tail does not invalidate the
structural checks on prior frames, but those frames still need independent
correlation with the actual trusted plan, model, input, execution status, and
scientific protocol before use. Recomputed checksums on modified files do not
establish provenance.

## Entry, failure, and cleanup

Keep a reference to both the plan and journal around an outer lifecycle guard.
Python can interrupt a context manager at its return boundary after hooks and a
lease have been acquired, so direct `with plan:` cannot by itself guarantee
cleanup after every asynchronous interruption. If entry was interrupted, the
same owning thread may call
`plan.recover_interrupted_entry(externally_quiescent=True)` only after the
caller independently establishes all of the following: the original entry and
body stack has completely unwound and cannot resume, no forward or hook is
running or can start, and no concurrent lifecycle operation is possible. An
idle but still active `with` body does not satisfy this precondition. The
method cannot infer these facts from Python flags. It aborts a recoverable
entered plan or clears its stranded entry marker and withholds public results.

If a forward, hook registration/removal, tree verification, or lease release
may still be active or uncertain, retain the plan, journal, and ownership;
retry `plan.close()` only after its documented quiescence preconditions hold,
or quarantine and discard the model/worker. `journal.close()` refuses to drop
its plan reference while known hooks, a plan transition, an entry marker, or a
held lease remain. A release interrupted after registry publication can leave a
stale held flag; the token-checked `plan.close()` retry resolves the local
state without releasing a newer owner. If cleanup also fails while another
exception is active, preserve the original exception and record the cleanup
failure as its cause. Arbitrary asynchronous cancellation and foreign code
cannot be made universally recoverable by this interface.

Read only after external process/forward quiescence and journal closure. A
failed close may be retried when the plan is safely cleaned up. A journal with
an indeterminate registration, active/uncertain forward, or unrecoverable
cleanup remains private and quarantined; a caller should discard that worker
rather than present the partial record as a completed experiment.
