# Synthetic causal v2: pre-seal bounded-failure clarification

This is a proposed clarification to the unsealed
[v2 protocol](CAUSAL_CONFIRMATION_V2.md) and the previously proposed
[patch-delivery clarification](CAUSAL_CONFIRMATION_V2_SELF_PATCH_CLARIFICATION.md).
Commit and independently review it before any v2 split or site schedule is
prepared. Bind this file's SHA-256 in a separate top-level
`resource_clarification_sha256` field of the public pre-manifest, without
changing the commitment `context` object's prescribed keys.

This clarification supersedes the earlier **64 MiB evidence limit** in the
proposed v2 and patch-delivery documents with a **1 GiB aggregate host-disk
budget** for study evidence. That is a storage allowance, not GPU VRAM or a
model-memory limit. The existing registry separately caps its declared
artifact payload at 512 MiB and 128 names; that implementation sublimit is
not a GPU constraint. The measurement on this host at 2026-10-29 01:56 UTC
reported 155,326,124,032 bytes free on C: (about 144.7 GiB), but free space
changes. Recheck the actual destination before the run and require at least
2 GiB free; also handle later disk exhaustion as an incomplete failure.

Use a **global fail-stop, no-retry** execution policy. The first unexpected
exception, failed required control, invalid metadata, failed source/numeric
check, storage failure, cancellation, or budget violation ends the attempt;
preserve already committed evidence and mark the attempt failed, cancelled or
incomplete as appropriate. Do not restart a failed attempt as though it were
the same run. A masked change, a no-difference observation, a zero effect, or
an abstention is a valid protocol outcome, not by itself a failure, provided
the injection-integrity, trace-completeness, metadata and control checks
required for that condition passed. A no-injection negative has no injection
integrity check; an active masked condition does. A missing event, invalid injection or failed control is not a
valid miss. The
planned successful schedule remains 57,408 forwards, with the hard cap of
60,000 attempted root forwards; the spare 2,592 is not a license to retry a
failed operation. Count an attempted root forward even if it raises. Also count
each intended forward submission before preparing its hooks or inputs; cap
submissions at 60,000, including any submission that fails before reaching the
model. Each submission may invoke the root at most once. Stop after the first
failed pre-forward preparation, so failures cannot repeat outside the counter.
This does not prohibit the original deterministic generator's bounded
collision-resolution attempts before model work; those are prescribed input
construction, not retries after an operational or control failure.

Fail-stop forbids another experimental forward or reuse of a contaminated
model; it does not skip the initial required hook and resource cleanup. Handle
`BaseException`, including `KeyboardInterrupt` and `SystemExit`. If hook
registration/removal, active-forward state or synchronization is uncertain,
retain or quarantine ownership and terminate/discard the worker; never reuse or
deep-copy a contaminated model. Preserve the original error when cleanup also
fails. If a fresh cancellation interrupts that cleanup, report the cancellation
as primary and retain the original error in its exception chain. If a terminal
record cannot be written, leave the attempt incomplete
and retain whatever evidence was successfully published.
If terminal publication raises, first verify the exact committed terminal
state and intended artifact hashes: an error can occur after publication. If
verification is unavailable or mismatched, report the outcome as unknown or
incomplete rather than assuming that the write failed. A crash or uncertain
publication never resumes the same attempt into held-out evaluation merely
because some freeze or commitment files exist.

Use straightforward bounded durable outputs that preserve completed records
and partial-failure evidence. One possible layout is one immutable chunk per
cell and phase (108 chunks), leaving 20 registry names for other evidence.
Specify bounded per-chunk records, single-writer ownership, duplicate
prevention, the state after interruption, and how an ordinary failure retains
the current cell's completed partial results. An elaborate journal or metadata
compression is not required. Flush and `fsync` each completed chunk before
acknowledging it, atomically publish it, and synchronize
the parent directory when supported. A crash before a verified terminal commit
leaves the attempt incomplete or unknown; completed chunks and any retained
staging file remain explicit and are not silently truncated or overwritten.
If an exact success outcome and
its intended artifacts were already committed, preserve that recorded success
while separately reporting the subsequent crash or acknowledgement/cleanup
failure. Complete required model and hook cleanup before publishing success;
neither case permits resuming the same attempt. These operations do
not promise survival of operating-system or power loss on every filesystem.
Cap untrusted read lengths
before variable-size allocation, then validate exact record counts, keys,
lengths and identities before serializing expanded records. Preserve hidden target details and held-out
outcomes in the private custodian record until the permitted reveal.

The public prediction freeze contains the frozen prediction functions,
selection results and protocol-defined discovery evidence, not concrete
held-out predictions. The concrete per-case record can reveal a private
input's polarity; write and synchronize its exact canonical bytes in private
custody before publishing its salted commitment. Publish only that commitment
before evaluation, and reveal the record and nonce only after the permitted
phase boundary. Neither private placement on the same OS account nor the
commitment proves secrecy from a noncooperating process.

The 1 GiB host-disk budget counts retained public and private evidence files,
the registry manifest and terminal outcome, and any simultaneously present
staging files or publication copies. Reserve space for terminal and
late-failure records before execution, and count linked file names
conservatively. The registry's `max_total_bytes` does not count its manifest
or outcome, so it is not by itself the aggregate disk check. Bound host-memory
buffers separately; neither limit is a GPU allocation limit. Check the actual
output layout and failure path using development data, including staging and
the registry envelope. Retain complete raw records when they fit; a synthetic
estimate is useful for planning but is not a guarantee that disk space cannot
change during execution.

If optional metadata templates or another compression scheme are used, version and
bind the codec before sealing. Validate exact recursive JSON types and
canonical bytes, rather than relying on Python equality that equates booleans
with integers or integers with equal floats. Require lossless reconstruction of
every actual metadata object, capture and source/base reference; reject a
mismatch before publication. Round-trip exact float32 bits, including signed
zero and subnormals. Never discard distinct evidence to fit the cap.

Define finite exact schemas and byte limits for public status and private
failure evidence before the run, including failures before a model forward and
failures after the last forward. Do not silently truncate an exception, omit a
required field, overwrite an earlier record, or turn an oversized failure into
a successful result. If the complete required error cannot be stored within
the reserved budget, record an evidence-limit condition where possible, retain
the existing partial evidence unchanged, and leave the attempt explicitly
incomplete; do not claim the complete exception was retained. If initial record
creation or the filesystem itself fails, complete failure evidence may not be
possible, and execution must not proceed. No local record establishes an
independent timestamp or a hostile-process security boundary.
