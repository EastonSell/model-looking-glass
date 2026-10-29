# Sampled evidence gate for grouped execution

`CertifiedBatchPolicy` is a trusted service-side gate for a planned grouped
forward. It neither executes a model nor schedules requests, reserves memory,
authenticates clients, or establishes universal row separability. A passing
sample says only that the evaluated mixed batches agreed with serial execution
within the frozen tolerances on those cases. Statistical independence and
generalization to other inputs do not follow from distinct hashes or IDs.

Before collecting evaluation results, the service must freeze and retain a
versioned protocol and its SHA-256 digest, tolerances, service maximum allowed
absolute/relative tolerances, and coverage threshold. Profile tolerances above
either service limit are rejected; choosing lax limits can still accept harmful
differences, so limits require independent scientific review.
It must independently retain the completed evidence digest. It supplies both
digests, the actual service-owned `BatchProfile`, and the separately verified
model fingerprint, checkpoint and adapter digests, and backend to the policy.
The profile's `runtime_id` must be `sha256:` followed by a separately retained
digest of a service-derived execution-environment manifest, computed with
`runtime_fingerprint` (canonical JSON with sorted keys and SHA-256). For CUDA this
manifest must include at least the physical GPU UUID, installed driver and
CUDA/PyTorch builds, backend, dtype, TF32/SDPA/autocast/determinism and other
relevant numerical modes; for CPU it must identify the physical/runtime and
numeric settings used. The service must construct and verify the manifest from
the actual executor and supply a newly collected manifest at each live-binding
check; the policy recomputes its digest. The service must use a versioned schema
and independently validate its collector and completeness. This helper accepts
generic manifests and does not verify their contents. A caller-supplied label
or a digest of an incomplete manifest is not verification.
The policy reconstructs the exact validator binding, makes its own JSON copies,
and calls `require_evidence` for each tested row count. Incomplete, failed,
tampered, mismatched, or untested evidence fails closed. CPU evidence cannot
authorize a different CUDA binding. Labels, hashes and assertions alone do not
authenticate the model, device, executor, or the truth of recorded runs; the
service must establish those independently, with ownership of protocol and
report data during their copies.

The service chooses `minimum_groups` in advance, at least two and at most 64.
For each tested combined row count, the policy requires at least that many
complete matching groups with distinct ordered row-hash sequences, at least
`combined_rows * minimum_groups` distinct full input-row hashes across those
groups, and at least two owners. `coverage(rows)` reports the number of complete
matching groups, distinct rows, and distinct owners. This is a conservative
coverage rule; even these requirements do not make cases independent or prove
correctness outside the sampled inputs. If no size meets it, construction fails.
`tested_rows` reports only admitted sizes; `allowed_request_counts` converts
those sizes using the exact per-request input shape. Repeated groups cannot
alone satisfy the distinct-group and row requirements.

`compatibility_key` binds the reconstructed profile, retained digests,
coverage/tolerance thresholds and admitted sizes into a 32-byte queue key. The trusted
coordinator still compares the live binding and each request again before
execution; a key is neither a capability nor a substitute for authentication.
`require_request` checks an exact forward request's seed, names, shapes, dtypes,
CPU tensors and no-gradient state. A mapping proxy is permitted only when it
comes from a trusted plain-dict snapshot; a proxy can otherwise wrap mutable or
custom backing code. These checks do not establish valid token, position, mask,
or segment values. The broker or adapter must validate model-specific domains
and exclusively own/lock input tensors and their backing mappings through worker
snapshotting, including against NumPy or `.data` aliases. One invalid input can
otherwise fail the entire batch.

This gate must be paired with a distinct, exact combined-shape workload profile,
whole-batch device admission, aggregate host-buffer custody, fair claim
recovery, atomic per-member cancellation/publication, synchronized execution,
and a worker that preserves identity and quarantines uncertain failures. Those
pieces are independent; the policy by itself does not enable GPU batching or
prove a throughput improvement.
