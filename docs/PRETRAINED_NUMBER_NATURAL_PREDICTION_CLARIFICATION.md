# Natural agreement v1: prediction record storage clarification

This pre-discovery clarification applies to the natural track of
`PRETRAINED_NUMBER_PATCH_V1.md` (SHA-256
`95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee`).
No natural intervention discovery or held-out forward has occurred when this
clarification is issued. The original protocol and the completed public
preflight remain unchanged. The scientific design, candidate selection,
predictions, baselines, controls, stopping rules and held-out procedure are
unchanged.

For the fixed random-site baseline, the protocol's Python byte literal denotes
the tag with hexadecimal bytes
`4d4c472d6e756d6265722d76312d72616e646f6d00`: the last byte is a NUL.
Appending the fixed seed in eight-byte big-endian form and applying the
specified SHA-256 rule selects zero-based index 5, layer 3 MLP, in the protocol's
ordered candidate list. This spells out the existing rule; it does not redraw
or change the baseline.

The protocol says that each discovery prediction file includes all discovery
measurements. For the natural track, this is satisfied by including every
derived site, family, target/donor, contrast and magnitude measurement in the
prediction record, and by including explicit SHA-256 links to the complete
audited discovery event log and independent numerical/landing audit record.
The event log indexes the actual saved tensors by SHA-256, dtype and shape.
The raw activation, logit and full-output witness bytes and per-forward control
records are retained as the referenced journal, rather than embedded in the
prediction JSON. Hashes alone do not reproduce the raw values: retain every
referenced payload and make the complete evidence available with the results.
Before publishing the prediction file or constructing the aggregate freeze,
complete the independent audit against the final retained journal and payload
bytes. Verify that its acknowledgment reports success and that its hashes
identify the final event log, journal start and terminal records, audit source,
and the payloads indexed by the event log. Retain the owner process's exit and
reaping evidence as well. If the journal or a referenced payload is
unavailable or fails audit, the track is incomplete and must not be
represented as a successful freeze. A later check does not retroactively
validate an earlier incomplete or unaudited freeze.

The canonical natural prediction schema is `mlg-number-natural-predictions-v2`.
In addition to the protocol/source/checkpoint identity, selection, ordered
present-tense and secondary past-tense site-metric events, it contains
`discovery_event_sha256` (the SHA-256 of the exact `events.jsonl` bytes) and
`independent_audit_sha256` (the SHA-256 of the exact saved independent audit
acknowledgment). Both are lowercase hexadecimal SHA-256 values. The canonical
JSON uses sorted keys, ASCII escapes, compact separators, finite numbers, and
no trailing newline. The audit record
binds the event log, journal start and terminal records, pinned preflight and
audit source; the event log binds each saved tensor payload. The aggregate
prediction freeze binds the SHA-256 of the exact canonical prediction JSON as
specified by the original protocol. It must be published before the held-out
nonce is drawn. The custodian retains the original journal, audit record and
all referenced raw bytes without overwriting failed attempts.

The natural track's source and runtime binding must also identify the exact
bytes of this clarification before discovery. A digest establishes a link to
bytes; it does not prove faithful execution, publication time, secrecy in a
shared account, or generalization to other inputs. Independent review must
still reconstruct the numerical effects, action landing and controls from
the retained raw evidence. This clarification adds no new claim about the
pretrained model and does not change the separate injection track's records.
