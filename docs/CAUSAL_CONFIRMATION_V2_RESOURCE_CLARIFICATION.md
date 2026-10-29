# Synthetic causal v2: pre-seal bounded-failure clarification

This is a proposed clarification to the unsealed
[v2 protocol](CAUSAL_CONFIRMATION_V2.md) and the previously proposed
[patch-delivery clarification](CAUSAL_CONFIRMATION_V2_SELF_PATCH_CLARIFICATION.md).
Commit and independently review it before any v2 split or site schedule is
prepared. Bind this file's SHA-256 in a separate top-level
`resource_clarification_sha256` field of the public pre-manifest, without
changing the commitment `context` object's prescribed keys.

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
fails. If a terminal record cannot be written, leave the attempt incomplete
and retain whatever evidence was successfully published.
If terminal publication raises, first verify the exact committed terminal
state and intended artifact hashes: an error can occur after publication. If
verification is unavailable or mismatched, report the outcome as unknown or
incomplete rather than assuming that the write failed. A crash or uncertain
publication never resumes the same attempt into held-out evaluation merely
because some freeze or commitment files exist.

Before sealing, specify and test a versioned, crash-safe journal/chunk layout
that preserves completed records and partial-failure evidence while respecting
the registry's maximum of 128 immutable artifact names. If it uses mutable
staging or an append-only journal, specify flush, synchronization, recovery,
single-writer ownership, duplicate prevention and the state after interrupted
publication. Flush and `fsync` each complete framed journal record or chunk
before acknowledging it, atomically publish completed chunks, and synchronize
the parent directory when supported. A process crash must not
be reported as a successful completion; any uncommitted tail remains explicitly
incomplete and is not silently truncated or overwritten. These operations do
not promise survival of operating-system or power loss on every filesystem.
Cap untrusted read lengths
before variable-size allocation, then validate exact record counts, keys,
lengths and identities before serializing expanded records. Preserve hidden target details and held-out
outcomes in the private custodian record until the permitted reveal.

The 64 MiB evidence cap is the sum of simultaneously retained serialized
evidence payloads and copies: public and private files, registry manifest and
terminal outcome, journal/staging payloads and memory-side publication buffers.
Reserve terminal and late-failure bytes before execution. Count file payload
bytes conservatively even when hard links may share physical storage. This is a bound on evidence payload,
not allocator peak, interpreter memory, filesystem block overhead or GPU
memory. The registry's `max_total_bytes` alone does not count its manifest or
outcome and therefore is not sufficient to enforce this aggregate cap. Measure
the actual journal and codec with complete synthetic serialized success,
partial and failure schemas, including simultaneous staging and verification
buffers, before sealing; an illustrative compression estimate is not a full
bound.

If metadata templates or another compression scheme are used, version and
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
