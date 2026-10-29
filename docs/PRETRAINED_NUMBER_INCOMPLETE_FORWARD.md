# Private incomplete-forward evidence

`encode_incomplete_forward` is a bounded in-memory serializer for raw objects
already available to a trusted injection custodian after a failed or uncertain
forward. It accepts an executor observation, its private action records and/or
one executor failure record. It does not invoke a model, validate the science,
establish execution or turn a failed forward into a completed one. Its bytes,
metadata, exception objects and decoded fields remain private until the
protocol permits reveal; they can expose the hidden site.

The binary record has a canonical bounded JSON header, byte payloads with
lengths and SHA-256 checksums, an overall digest and a footer. Tensor payloads
are copied as bytes, including nonfinite float32 bit patterns and signed zeros;
no pickle or floating-point conversion is used. Repeated references to the
same immutable byte object are stored once within a record. The record also
preserves bounded action/trace coordinates, source identities, private metadata,
available failure captures and logits. The process-local owner token is
deliberately excluded. Exception text may already have been clipped by the
executor and is marked accordingly.

The raw tensor/evidence byte allowance is 1 MiB, text blobs have 128 KiB, and
the header has 128 KiB; framing can therefore exceed 1 MiB even though the
raw tensor cap is respected. The format reports individual unavailable,
invalid, oversize or over-budget fields in `omissions`; it never silently
truncates a byte field. Absent observation, action, failure or capture data is
represented explicitly. Failure snapshots discarded by the underlying plan
before reaching the executor are unavailable and cannot be reconstructed.
Metadata and raw bytes are not authenticated or proven to come from a model by
their hashes. A failed serializer, interruption, OOM or failed disk write can
prevent a complete record; the supervisor must retain any other available
evidence and independently report which pieces are unavailable, without retrying
the failed study forward.

`decode_incomplete_forward` validates the bounded framing and hashes and
returns private payload bytes and their manifest; decoding does not make the
record a successful observation. The caller must separately reserve the
retained source objects, the encoded copy, decoder/serialization temporaries,
filesystem overhead and physical disk use. The protocol's 512 MiB cap on
logical retained tensor payload is study-wide, not a per-component allowance;
the supervisor must reconcile all tracks, failures and orphan records.
If an extreme malformed input has more than the bounded number of distinct
omissions, the final `additional-omissions` marker records that the list itself
is incomplete.
