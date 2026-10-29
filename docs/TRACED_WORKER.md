# CPU-only traced GPT-2 worker

`TracedGPT2Worker` connects the resident worker and request-local capture
dispatcher for a trusted owner. It runs an existing, preconstructed
`GPT2LMHeadModel` on CPU; it does not download weights. CUDA tracing is
deliberately rejected until device-memory admission and trace ownership are
integrated and verified. This is not a GPU throughput result, a public broker,
a general model adapter, or a causal explanation.

The model must be in evaluation mode, use standard GPT-2 block, attention and
MLP types without aliases, have caching and gradient checkpointing disabled,
and be exclusively owned by the worker. `past_key_values` and cached decode
inputs are rejected. After a failed or cancelled forward, do not retry an
external mutable cache whose consistency has not been verified; re-prefill or
restore it from a validated snapshot. The owner selects nonempty
`TraceSite(layer_index, module_kind)`
tuples from the fixed `block`, `attn`, and `mlp` boundaries. No submitted
function, hook, arbitrary module path, or model code is accepted as a trace
plan. Exactly one root forward and at least one observation of every requested
site are required. The bridge extracts a direct tensor or the first tensor in
a tuple/list and records the actual output selector. Hooks return `None` and
never replace a model output.

The trusted caller submits an ordinary `ForwardRequest`, one or more
`CaptureRequest`s, explicit run-scoped `sequence_starts`, and per-batch event
and retained-tensor-byte caps. The dispatcher additionally enforces each
request's event and byte limits. Inputs are snapshotted before their shape is
used for dispatch, then snapshotted again by the resident worker; limits do
not count these extra copies, metadata, model activations, temporary slicing,
or results held by earlier consumers. The owner must enforce any aggregate
host-memory retention and release policy separately.

**Trust boundary:** the caller must verify that each declared request/sample
owns its input row. The dispatcher checks that rows have exactly one declared
owner, but cannot prove that those declarations are true. A successful
`TracedForwardResult` contains **all** logits for the combined batch and all
request-owned `CapturedSlice`s. It must be returned only to the trusted batch
owner, which must route authorized logits rows and slices to each requesting
team. It must not expose this whole-batch object to an individual client.
Per-event IDs and coordinates do not themselves provide access control.

`CapturedSlice.to_trace_event()` emits validated immutable metadata, while
`.value` is an independently owned mutable CPU tensor. Saved shape omits the
selected source batch axis. Batch and token positions refer to the original
hook output, not model-wide or cache positions. Generation stage and cache
fields remain unknown; even a supplied generation step does not assert cache
semantics. `sequence_starts` and `forward_index` are supplied by the owner;
cross-call uniqueness is not allocated or enforced here. The owner must use
distinct run IDs or maintain these counters consistently across calls and
processes. Matching positions or tensors does not establish causal provenance.

The outer worker gate serializes traced and untraced calls made through this
subclass. A nonblocking process-local `ModelLease` is acquired before resident
worker execution and held through hook registration, forward/device
completion, hook removal, state checks, and cancellation checks. The ordering
is outer gate, model lease, resident worker lock, dispatcher execution lock,
and worker RNG lock. The model lease is not an interprocess lock and cannot
guard code that ignores it. Model construction and external mutation remain
trusted, as in the resident worker's existing contract.

Hooks are installed inside the resident worker lock and removed in `finally`,
including partial registration failures. Capture output stays private until
the normal worker's output, model-state, poison, and late-cancellation checks
pass. Cancellation, failed capture, OOM, bad output, or failed postchecks yield
no trace result; the underlying worker's poisoning policy remains in force.
If hook registration fails before returning a handle, hook removal fails, a
hook remains attached, model-tree verification fails, or registered model state
changes or cannot be verified, ownership is quarantined and the lease and
model references remain held until process teardown. A new owner must not reuse
that uncertain model. Exception tracebacks
are sanitized to avoid retaining private inputs or traces; bounded failure
type/message diagnostics are preserved, including an initial failure when
cleanup also fails.
If lease release fails or is interrupted, the bridge retries it once using the
lease's token-checked, idempotent release operation. A second failure poisons
the worker and retains the lease handle for diagnosis. That uncertain release
may have happened before or after registry publication, so the retained handle
alone does not establish whether the model is still reserved.
The existing `ResidentWorker` and its untraced behavior are unchanged.
