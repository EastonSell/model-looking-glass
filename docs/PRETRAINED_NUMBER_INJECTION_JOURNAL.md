# Private injection discovery journal

This CPU-only, single-owner component stores paired public observations and
private intervention witnesses for the **discovery** schedule. It does not
launch a model, publish a request plan, choose a secret site, authorize
held-out data, or establish a completed study. Use only synthetic data until a
separate reviewed supervisor establishes all protocol gates. The protected
records, directory inventory, receipts, exception objects and tracebacks can
reveal the injection site. Keep them in trusted custody until the protocol
permits reveal; they are not selector-facing output.

The caller supplies the exact separately published source commit, request-plan
bytes and digest, runtime digest, site commitment and private reveal inputs.
The helper recomputes the request plan and verifies the commitment, then binds
those identities and the protocol/checkpoint revision to each private frame.
The site layer and nonce are not included in the binding. Neither the binding
nor a checksum establishes that the declared source actually ran, that the
runtime/model was correct, that publication happened, or that public entropy
was drawn independently before the site nonce. The supervisor must establish
those facts, and must not expose frame hashes or failure details as a proxy for
the hidden site. Exact equality between the site nonce and the published
request entropy is rejected; inequality alone does not establish independent
randomness. A later held-out split nonce also needs independent generation and
comparison against both previously used values.

For each of the 128 precommitted requests, `append` exclusively creates a
checksummed pair frame containing the bounded canonical public observation and
private completed-forward record. It writes and fsyncs that **unverified**
frame before parsing and auditing it, so a semantic failure does not silently
discard the available bounded pair. The audit verifies the planned ordinal,
case, role, source/request IDs and same-case clean/injected baselines, then
checks the private witness. Only after audit does it exclusively write and
fsync the acknowledgment marker and return a private `PairReceipt`. One pair
counts as acknowledged only if the supervisor independently receives and
retains that exact receipt. A frame or marker alone cannot establish that its
write or fsync was acknowledged, that stable media persisted it, or that any
forward actually occurred.

The journal requires the family-final v2 request-plan schema. Each family's
two cases complete their 15 non-final roles before either `verify_clean` is
accepted. Its baselines stay keyed by case through both final clean checks;
the earlier v1 case-major schedule is rejected rather than reinterpreted.
For a combined natural/injection family, the supervisor must also place these
checks after that family's natural interventions if they are to serve as shared
family-final controls; the injection journal cannot establish cross-track order.

Any in-method error, cancellation, audit rejection or budget overflow poisons
the writer. It does not retry a request or resume an attempt. A best-effort,
bounded `terminal.json` contains only a fixed phase, failed/cancelled status,
and a conservative statement that exception details are omitted or may have
been clipped. Terminal write failure can leave no terminal file; the supervisor
must independently preserve a sanitized terminal status and any available
evidence. If a forward fails before a completed pair exists, `abort` can retain
caller-supplied bounded opaque private partial bytes in a distinct incomplete
file, or state that evidence is unavailable. It does not claim the activation
plan supplied snapshots that were discarded. The supervisor must preserve the
original exception and treat the whole attempt as non-retryable even if failure
reporting also fails.

Opaque partial bytes are limited to 1 MiB. The separately validated
`pretrained_number_incomplete` format can use a larger bounded envelope for
metadata and framing while retaining at most 1 MiB of raw tensor bytes. The
journal charges the entire envelope to its local reservation, and inspection
revalidates oversized envelopes. A serializer failure still requires the
supervisor to report unavailable pieces rather than inventing a record.

The owner is the actual creating `Thread` object. A foreign thread cannot
append, and concurrent use is outside the contract. Python can interrupt at a
call or return boundary outside the method's handler. If the caller cannot
prove that the writer and forward have both stopped and that it observed the
receipt, it must retain the private directory as incomplete and discard the
worker/journal; it must not infer success from the on-disk marker.

`inspect_discovery_journal` requires caller-established quiescence, the same
trusted bindings, ordered independently observed receipts, and a read limit.
It rechecks the hash chain, exact allowed directory inventory, each receipt,
the planned order and private semantic controls. It returns only the verified
prefix and explicitly distinguishes an unacknowledged pair, a partial record,
invalid bytes, capacity and an empty tail. A structurally complete but
unacknowledged pair can be returned privately as **unverified** evidence.
`complete-prefix` means all 128 pairs passed those checks; it is not a claim of
model execution, publication chronology, stable-media durability, successful
cleanup, prediction freeze, or held-out success. It does not replace the
selector's public archive audit or the supervisor's whole-study coverage and
source/model/runtime checks.

Before asserting private controls to the selector, the supervisor must also
check model/buffer and execution settings before and after, per-forward hook
cleanup, successful executor exit, the family-final controls, every
independently retained receipt, and the complete public archive recomputation.
An incomplete or uncertain check leaves the attempt incomplete.

The attempt directory must be a new absolute path under a trusted parent with
an appropriate user-controlled ACL. Exclusive creates prevent accidental
overwrite but do not defend against a hostile or concurrently replaced parent,
symlinks elsewhere in the path, or another process with filesystem access.
The requested `0o700`/`0o600` modes are best-effort POSIX permissions and do
not configure Windows ACLs. The caller must ensure exclusive trusted model and
filesystem use throughout; before/after state checks do not rule out transient
runtime mutation.

The disk allowance reserves both frame and acknowledgment sizes for every
attempt plus 4 KiB for a terminal record, without refund after failure; it
covers logical file bytes, not filesystem overhead. The retained-payload
reservation conservatively charges the full encoded pair or opaque partial
bytes without deduplication. Those lengths bound the represented tensor bytes
but can overcount them; the allowance is local to this journal. The supervisor
must combine it with existing executor-owned snapshots, other tracks,
failed/orphan records and held-out work under the protocol's single 512 MiB
study-wide limit. The encoded per-pair limit is 1 MiB, also conservative for
the raw per-forward payload. Reading has its own encoded-byte cap. Python
decoding, temporary copies, staging buffers, model parameters, allocator and
filesystem caches, and process RSS are separate and need explicit headroom.
The protocol's GPU/VRAM admission and isolated benchmarks are outside this
component.
