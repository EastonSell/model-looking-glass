# Synthetic v2 evidence: development-only storage helper

This helper stores fabricated or future study records; it does not run a model,
generate held-out cases, enforce the discovery/held-out gate, or prove a causal
claim. The protocol owner must validate its exact case, trace, plan, identity and
record-count schemas before marking a chunk complete. A chunk's status is a
caller assertion, not proof that every required computation ran.

The trusted owner creates a new run with `create_evidence`, passing the frozen
protocol/input/model/code identities, one byte limit per cell/phase chunk,
limits for any other artifacts, and a nonzero private failure-artifact limit.
The registry permits at most **128 declared names and 512 MiB total declared
artifact payload**; those implementation limits are separate from the v2
protocol's **1 GiB aggregate host-disk evidence allowance**. Neither number is
a GPU/VRAM limit. The constructor checks at least 2 GiB free on the destination
filesystem immediately before creation; this is a snapshot, not a reservation.

The conservative disk bound adds every declared artifact limit, the larger of
one maximum-sized artifact or one maximum-sized registry metadata record for
the `.pending` path, the registry's maximum manifest and outcome sizes, and
`external_reserve_bytes`. It counts the pending and published hard-link paths
separately even if they share physical blocks. Only ordinary files in the
registry run are accepted; unrecognized paths or filesystem links fail closed.
The caller must independently ensure that all public/private evidence retained
outside the run fits the declared external reservation, and must coordinate
all writers so that they cannot bypass the helper. The helper does not inspect
external files or bound Python object memory, JSON parsing/serialization
copies, OS caches, filesystem block overhead, or allocations by other programs.
The runner must measure and bound its own peak host memory and all record sizes.

For each expected name, `publish_chunk(name, canonical_json_bytes)` accepts a
compact sorted-key UTF-8 JSON object with exactly these fields:

```json
{"name":"example.json","records":[],"schema":"causal-v2-chunk-v1","state":"complete"}
```

The other allowed state is `incomplete`. An ordinary experimental failure can
publish an incomplete chunk containing all completed records buffered for that
cell, then publish a failure and a failed/cancelled outcome. Success requires
every declared chunk, its `complete` status, and every other required artifact.
The caller must enforce semantic completeness, exact recursive JSON types,
source/condition identities, and bit-preserving float32 representation; this
generic envelope checks none of those properties inside `records`.

`publish_failure` requires canonical private JSON with exactly `schema`
(`causal-v2-failure-v1`), `outcome` (`failed` or `cancelled`), `stage`,
`error_type`, `message`, `submissions` and `root_forwards`. Optional
`cell_id` and `phase` are opaque strings or null. Counts are exact integers
with 0 ≤ root forwards ≤ submissions ≤ 60,000. The complete sanitized message
must fit the declared limit; it is never truncated. Keep private diagnostics in
the private run. The caller supplies a separate generic public status and must
not expose private fields to an investigator before the protocol permits it.
A failure outcome must agree with the failure record's outcome. If a complete
failure cannot be encoded or stored, preserve existing artifacts and report the
run as incomplete rather than inventing a complete error.

Every artifact is staged as `.pending`, written fully, flushed and file-fsynced,
then published with a no-replace hard link. On POSIX the helper attempts to
fsync the parent directories; Windows has no portable directory-fsync path
here. A failed write retains any pending bytes. A process crash leaves already
published chunks intact; a crash during the current chunk can leave an
uncommitted partial `.pending`. A crash after linking can leave both paths.
`status()` reports pending size, digest and any hard-linked artifact or
outcome; these facts do not establish that the caller observed acknowledgment.
No silent truncation, overwrite, or automatic retry is performed. The
filesystem must support the documented local lock and hard-link semantics; the
protocol does not promise survival through all power losses or storage devices.

Only the handle returned by `create_evidence`, in its creating process and
thread, can write. A reopened handle is inspection-only. Exceptions while
mutating poison the writer, including ambiguous publication or lock-release
errors. A validation rejection before mutation permits only a bounded
failure-only path, then termination. Arbitrary asynchronous exceptions at a
Python return boundary cannot always be observed inside the callee: the caller
must treat **every** interrupted or uncertain operation as fail-stop and never
continue the same attempt. Inspect the exact intended bytes and published hashes
to reconcile a post-publication exception; an unresolved outcome is incomplete.
Using the underlying registry directly, modifying files outside these locks, or
rewriting a self-consistent run is outside this cooperative trust boundary.
The registry outcome hashes the artifacts and manifest, but supplies no
external timestamp, signature, access control, or proof of execution order.
