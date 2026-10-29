# Trace metadata schema 1.0

`looking_glass.trace_schema` defines immutable, metadata-only observations. It
does not capture tensors, schedule work, impose a memory cap, or establish that
an activation caused a behavior. Its `TraceEvent` is separate from the original
`looking_glass.probes.TraceEvent`; the existing probe API and its results are
unchanged. Producers must explicitly adapt their observations to this contract.

## Identity and ordering

- `RunIdentity.run_id` is a nonempty producer-assigned ID for a trace run.
  IDs must be unique within the storage or analysis namespace. Optional
  `job_id`, `request_id`, and `sample_id` are nonempty IDs or `None` if unknown.
  A batch event must not claim a single sample identity unless the producer has
  actually associated the saved slice with that sample. Batch row numbers are
  not stable sample IDs, including during beam reordering.
- `sequence` is a nonnegative integer assigned once per emitted event within
  a run. **Every separately emitted sample slice gets a distinct sequence**,
  even if slices came from the same hook invocation. The producer is responsible
  for coordinating this allocation across batches and concurrent producers.
  `(run_id, sequence)` is the event identity. `validate_trace(events)` detects
  duplicates within the supplied collection; it cannot detect collisions in
  files or streams that were not supplied. Partial or reordered collections
  need not have contiguous or sorted sequences. Sequence values identify events;
  they do not guarantee chronological order across independently allocating
  producers unless a separate producer ordering contract says so.
- `forward_index` is the zero-based index of a root forward call within the
  run. A run may span multiple forward calls and requests. Within a run, slices
  of the same physical root call share a `forward_index`, and distinct root
  calls have distinct indices; producers coordinate this assignment. When a
  physical batch serves distinct runs, its events can have different indices
  according to each run's history. `invocation` counts calls to the same
  canonical `module_path` within that forward, starting at zero. Multiple slices
  of the same call share these coordinates while having separate event
  identities. `module_kind` is an
  optional producer-defined label such as `block`, `attn`, or `mlp`;
  `layer_index` is an optional nonnegative architecture-specific index. A path
  or layer number is meaningful only alongside the separately recorded model
  identity, revision, and instrumentation configuration. `forward_index` and
  `invocation` are required known coordinates: if a producer cannot determine
  them, it must not invent zero or emit a schema 1.0 event.

## Tensor coordinates and storage

`shape`, `dtype`, and `device` describe the original tensor output observed at
the hook, before selection or offloading. `saved_shape`, `saved_dtype`, and
`saved_device` describe a retained snapshot when known; any may be `None` when
unknown or unavailable. They do not attest that a payload is still available.
The payload, if any, is managed separately. `capture_mode` is a nonempty
producer-defined label, for example `direct`, `gradient`, or `id`; the schema
does not infer properties of an unfamiliar label.

`output_path` identifies the selected tensor inside the module's output:
`()`: the output itself is the tensor; an integer selects a tuple or list
element; a string selects a mapping key; and `None` means the selection is
unknown. For example, `(0,)` selects the first tuple component and
`("hidden", 0)` selects the first item of the mapping's `hidden` value. Other
container access conventions must be documented separately; do not represent
an attribute access as a mapping key unless the output actually supports that
mapping operation. This selector prevents confusing different tensors emitted
by the same module invocation; no validity check can inspect the original
runtime output after the fact.

`batch_positions` and `token_positions` are ordered tuples of zero-based
indices into the **original hook output's** local batch and token axes. They
describe the selection, so duplicates and empty tuples are representable.
`None` means the selection is unknown; it does not mean an empty selection or
the full axis. `batch_axis` and `token_axis` identify those axes in `shape` when
known, and must be distinct. Known axes and their indices are range-checked.
When an axis is unknown, the schema cannot check index bounds. Token positions
are not automatically positions in an input prompt or generation cache: during
incremental decoding, local token index 0 may correspond to a later model/cache
position. Neither matching coordinates nor matching activation values establish
identical input tokens, identical models, causal provenance, or a computation's
purpose.

## Generation and caching

`GenerationContext.stage` is one of `unknown`, `forward`, `prefill`, or
`decode`, supplied by the producer. `generation_step`, `cache_position`,
`cache_length_before`, and `cache_length_after` are optional nonnegative
integers; `use_cache` is an optional Boolean. They are never inferred from local
token indices or tensor shape. A scalar cache position is only appropriate when
one value accurately describes the event; otherwise use `None` and preserve
the model-specific details in the experiment manifest. The producer must record
the model's generation and cache conventions separately. Cache lengths need not
increase, since some models evict or truncate cache entries.

## JSON format and validation

`TraceEvent.to_dict()` produces a fresh JSON-compatible object;
`TraceEvent.to_json()` emits a deterministic compact JSON object. The wire
object has `schema_version: "1.0"` and contains all fields, with `null` for
unknown values. Each event can be stored on a separate JSON Lines line.
`from_dict` and `from_json` strictly require this version and all defined fields,
reject additional or missing fields, duplicate JSON keys, nonstandard numbers,
invalid types, negative coordinates, and out-of-range known axes/positions.
Tuples in Python are JSON arrays on the wire. Rejecting unknown fields and
versions prevents silently misinterpreting a future schema; any migration must
be explicit. Validation is local to the declared metadata: it cannot verify a
producer's claimed axis semantics, sample mapping, model, or payload.

Example:

```python
from looking_glass.trace_schema import GenerationContext, RunIdentity, TraceEvent

event = TraceEvent(
    identity=RunIdentity("run-1", "job-1", "request-1", "sample-1"),
    sequence=0, forward_index=0, module_path="transformer.h.1.attn",
    invocation=0, module_kind="attn", layer_index=1, output_path=(0,),
    shape=(4, 8, 64), dtype="torch.float32", device="cuda:0",
    batch_axis=0, batch_positions=(3,), token_axis=1, token_positions=(2, 7),
    generation=GenerationContext(stage="forward"), capture_mode="direct",
    saved_shape=(1, 2, 64), saved_dtype="torch.float32", saved_device="cpu",
)
assert TraceEvent.from_json(event.to_json()) == event
```
