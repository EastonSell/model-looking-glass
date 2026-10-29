# Experiment registry v1

The registry provides local, append-only run manifests, bounded artifacts, and a
single terminal outcome for cooperating processes. It does not schedule GPU
work or automatically register the repository's earlier experiments; their
existing result directories remain unchanged. The registry stores evidence,
not a causal explanation.

## Identity and creation

`create_run` exclusively creates the specified directory. Its parent must
already exist. A pre-existing directory is never reused. The versioned manifest
contains `protocol`, `inputs`, `model`, and `code` JSON objects and a SHA-256
digest of the canonical JSON bytes of each object. It also contains run metadata,
declared artifact byte limits, required artifacts, and the total artifact budget.
Canonical bytes are UTF-8, sorted-key, compact JSON produced by Python's JSON
encoder, with finite numbers and string keys. This is this schema's encoding,
not a claim of RFC 8785 or a universal cross-language encoding.

The registry hashes the objects the caller supplies: **a hash of a model name
is not a hash of its weights**. Callers must include actual content hashes or
immutable, independently resolvable identities for the protocol, input data,
checkpoint, tokenizer, code, and other output-relevant components. Include
model and tokenizer revisions, seeds, runtime/backend settings, intervention,
capture policy and splits in the identity objects; reserve metadata for labels
that do not affect outputs. In particular, record the content or reproducible
identity of every held-out case. The registry accepts empty identity objects and
does not validate protocol-specific completeness. The four identity digests are
not a safe deduplication key until the caller has independently validated that
every output-relevant input is represented and recoverable. The full manifest
hash also covers metadata and budgets, but matching it alone does not prove the
supplied identities actually identify their underlying data.
`sha256_file(path)` hashes file contents incrementally, but cannot decide which
files or settings identify an experiment. The registry does not resolve remote
identifiers or deduplicate work.

Each concrete protocol should specify and validate its minimum provenance:
frozen protocol version and content; input, split and case identities; model
configuration, weights and tokenizer; executable source revision plus a hash
or archived copy of uncommitted changes; seeds; device, dtype, backend and
dependency versions; and intervention and capture settings. Record parent-run
and retry lineage, when applicable, alongside failure and cancellation details.

For example (replace the sample hashes with measurements of the actual files):

```python
import json
from looking_glass.registry import create_run, verify_run

run = create_run(
    "results/study-run-001",
    protocol={"id": "intervention-study-v1", "document_sha256": "actual-hash"},
    inputs={"discovery_sha256": "actual-hash", "held_out_sha256": "actual-hash", "seed": 17},
    model={"checkpoint_sha256": "actual-hash", "tokenizer_sha256": "actual-hash"},
    code={"commit": "actual-commit", "source_sha256": "actual-hash"},
    metadata={"owner": "study-team"},
    artifact_limits={"precommit.json": 65536, "results.json": 65536},
    required_artifacts=["precommit.json", "results.json"],
    max_total_bytes=131072,
)

def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")

# Freeze all predictions and effect definitions BEFORE any held-out forward pass.
run.write_artifact("precommit.json", encoded({"predictions": []}))
# Evaluate the prespecified held-out cases and preserve their individual outcomes.
run.write_artifact("results.json", encoded({"cases": []}))
run.finalize("succeeded", {"protocol_completed": True})
assert verify_run(run.path)["state"] == "succeeded"
```

The placeholders and empty arrays above demonstrate the interface, not a
valid study or result. The caller controls and must enforce evaluation order;
the registry cannot establish that predictions preceded model execution.

## Lifecycle, crashes and errors

`open_run(path)` opens an initialized run from another process. A run can
publish each declared artifact once, as `bytes`, using
`run.write_artifact(name, data)`. A success can be finalized only when every
required artifact exists. `run.finalize(state, details)` accepts `succeeded`,
`failed`, or `cancelled`; it atomically commits exactly one outcome, including
the manifest SHA-256 and each present artifact's size and SHA-256. Failed or
cancelled runs may retain any already published precommitment and partial
evidence. Once terminal, later writes and terminal outcomes are rejected.

The runner must catch errors and explicitly record a failure or cancellation;
the registry does not infer either from a vanished process. If writing that
record fails, preserve and report the original error and the recording error.
Do not invent a successful summary. An initialized run without an outcome is
`incomplete`: it might still be active or have stopped unexpectedly. A directory
without a manifest is `uninitialized`; if it contains a published outcome or
artifact, verification instead reports corruption. An interrupted initialization
may leave an uninitialized directory, so use a new run directory rather than
reusing it. `verify_run` checks manifest structure and identity hashes; for a
terminal run it also checks the manifest and the exact artifact inventory
against the outcome. For an incomplete run it reports current artifacts, but
there is no terminal digest against which to verify their previous contents.

Each operation takes a per-run, OS-backed lock shared across cooperating
processes; process termination releases it. While holding that lock, a writer
uses a single scratch file, flushes and `fsync`s it, then publishes with a
same-filesystem hard link that cannot replace an existing target. This requires
a filesystem supporting those lock and hard-link semantics; failure is surfaced,
with no unsafe overwrite fallback. A killed writer may leave `.pending`. The
next write or finalization removes that scratch file while holding the lock,
and status reports `pending_write` if it remains. Once the hard link succeeds,
an operating-system error removing `.pending` does not turn the published
record into a failed write; the leftover scratch file is reported by status and
the next locked mutation retries its removal. If publication itself fails and
scratch cleanup also fails, the publication error remains primary and the
cleanup error is chained. Other system errors or interruptions after publication
(for example, while releasing the lock) can still make the caller uncertain
whether a write or finalization succeeded: inspect the run and compare its
published contents with the intended operation before deciding what happened.
Atomic visibility and file `fsync` do not
promise durability through every power loss, controller, filesystem, or network
share; test the deployment filesystem and keep independent backups as needed.

## Limits and trust boundary

The sum of per-artifact declared limits must fit `max_total_bytes`. V1 accepts
at most 128 artifacts and a total limit of 512 MiB; manifest and outcome JSON
are separately limited to 1 MiB each. Only safe ASCII basenames are accepted,
including Windows reserved-device and case-alias checks. An undeclared,
oversized, or repeated artifact is rejected. The one scratch file is also
bounded by the artifact or metadata currently being published; hard links
refer to the same file data rather than duplicating the contents. Callers still
own any memory they allocate before passing a `bytes` value.

These guarantees apply to callers using this API and cooperating with its locks.
The files are not write-protected against another local process or person.
Terminal verification detects changed or missing recorded artifacts or manifest
against the stored outcome, but a party able to rewrite the entire directory
could forge a consistent set of hashes. Deleting the outcome alone makes a
previously terminal run indistinguishable from an incomplete one unless an
independently protected record says an outcome existed. No signature, access
control, encrypted storage, privacy filter, or external timestamp is supplied.
Do not put secrets, private prompts, or sensitive activations into shared/public
artifacts without an appropriate data-handling policy.
