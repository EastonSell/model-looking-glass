# Comparing observed activation traces

Choose the comparator for your event type: `compare_traces` handles the original
probe events, while `compare_transformer_traces` uses schema-v1 metadata and
explicit transformer coordinates. Both report observed differences, not causal
origins.

## Original probe events

`looking_glass.localization.compare_traces` compares the **original**
`looking_glass.probes.TraceEvent` streams. It matches a named tap and its
per-tap occurrence ordinal explicitly; it never assumes that the *n*th value in
one stream corresponds to the *n*th value in the other. The current function
does not accept the newer transformer event schema: use an explicit,
coordinate-preserving adapter before comparing those events. In particular,
node-and-occurrence labels alone do not encode layer, token, output selection,
forward index, or generation/cache context.

```python
from looking_glass.localization import compare_traces

report = compare_traces(
    reference_session.events, changed_session.events,
    atol=0.0, rtol=0.0,
    reference_complete=True, candidate_complete=True,
)
print(report["status"], report["first_observed_interval"])
```

Only set each completion flag if that stream finished successfully and includes
every emitted event under its declared capture policy. The default is `False`.
This attestation covers the declared taps, not the whole model. An omitted
module, hidden branch, unrecorded token, truncated execution, or discarded
artifact is not negative evidence. The caller is responsible for checking model
and code revisions, selected inputs and tokens, instrumentation policy, runtime,
and other experimental conditions; matching event labels do not establish any
of these.

The JSON-compatible report contains the chosen tolerances, individual matched
comparisons, both streams' counts and completion flags, run-identity consistency,
sequence and occurrence continuity, missing and duplicate coordinates, full
coordinate-order agreement, unavailable values, and incompatible tensor metadata.
Missing values, nonfinite or empty tensors, unsupported layouts or devices,
mismatched shape/dtype/device, and inconsistent saved-value metadata cannot
establish equality. Empty streams cannot establish equality either. Only dense
strided CPU or CUDA tensors in float16, bfloat16, float32, or float64 are
supported; sparse, meta, float8, named tensors, lazy negative/conjugate views,
and other values are reported as unavailable. Values are compared elementwise
against the reference magnitude:
`abs(candidate - reference) <= atol + rtol * abs(reference)`. This inequality
uses exact real arithmetic over the recorded finite IEEE tensor values and the
binary64 tolerance values. It can therefore differ at a boundary from the
previously implemented rounded binary64 computation. Signed zeros compare equal.
Both tolerances must be finite, nonnegative built-in Python `int` or `float`
values; integers are converted to binary64 first. Other numeric types must be
explicitly converted by the caller before use. Restricting the conversion avoids
silently flushing a subnormal in an implicit conversion; integers that are not
exactly representable as binary64 are rejected instead of rounded. Zero
tolerance is the default. Subnormal bits are preserved during comparison even if the calling
CPU thread flushes denormals; this does not establish the numerical mode used
when the model produced the retained values.

The `tolerance` report includes `rtol_binary64_hex` and `atol_binary64_hex`,
each exactly 16 hexadecimal digits encoding the authoritative IEEE-754
binary64 bits. These are bit patterns, not Python's `float.hex()` strings.
The corresponding numeric `rtol` or `atol` field is `null` for subnormal
tolerances in either CPU mode: Python JSON formatting can otherwise print a
retained subnormal as zero while denormal flushing is enabled. Other finite
tolerances keep their numeric fields, and the hex fields preserve signed zero.

- `difference_observed`: a difference is present at a matched observation and
  all earlier supplied observations form an aligned, available, equal prefix.
  Both completion flags must be explicitly true. `first_observed_interval`
  reports the immediately preceding equal tap, if any, and the unequal tap.
- `no_difference_observed`: both streams were explicitly declared complete,
  contain at least one comparable value, and every event aligned and compared
  equal within tolerance. This is a result about those captured values only.
- `inconclusive`: neither conclusion is supported. A later matched difference may
  still appear in `first_compared_difference` (in reference-stream order), but an unmatched, ambiguous,
  reordered or unavailable earlier event makes the first-observed interval
  unknown.

An occurrence is an ordinal in the observed stream. If a repeated module call
was skipped or changed semantically, equal ordinals need not refer to the same
computation. Even for matching events, the final equal observation and next
unequal observation are sample boundaries, not a proof that the intervention's
cause lies between them: effects can appear and disappear between taps or travel
through unobserved paths. `causal_origin` is always reported as `"unknown"`.
Establishing or rejecting a causal explanation requires separately controlled
interventions and held-out evaluation; activation differences alone do not
explain model reasoning.

This comparator does not save tensor values, impose a trace-memory budget, or
schedule GPU work. Finite-value checks and element comparisons can synchronize a
GPU. Exact comparisons use temporary integer views and masks, and nonzero
tolerances stage complete saved snapshots on CPU before processing bounded
chunks of Python integers. A noncontiguous value can require a full-size
contiguous copy, including a device-side staging copy for CUDA. The caller must
budget source-sized device and host copies and use supervised GPU ownership;
the integer chunks do not bound this snapshot staging. Nonzero-tolerance
comparison may be substantially slower than vectorized float comparison.
Measure or budget these costs separately, and prefer bounded CPU-staged traces
when appropriate. No new empirical localization result is claimed here.

## Transformer observations

`looking_glass.transformer_localization` compares tensor payloads attached to
`trace_schema.TraceEvent` metadata. It uses the existing conservative
localization comparator after encoding explicit source coordinates. It does not
automatically accept or infer metadata from an arbitrary model hook.

### DistilBERT encoder-forward comparisons

For results produced by `trace_distilbert_masked_lm`, callers may pass
`encoder_contexts=(reference.encoder_context, candidate.encoder_context)` and
an explicit `expected_input_differences` tuple to `compare_transformer_traces`.
Use `()` for identical input IDs, or list exactly the changed `(batch, token)`
positions. The contract requires identical complete masks and input shapes,
checks all input ID differences against that exact set, and verifies the full
selected layer/module/sample/token event policy, run identity, order, forward
and invocation metadata on each side. It checks recorded model dimensions;
it does not authenticate model weights, runtime, provenance, or the truth of a
producer-supplied context. Local token coordinates are not semantic alignment.
Keep the explicit completion, pairing and execution-order attestations: the
caller must verify model and runtime correspondence, tokenizer/semantic
alignment and the producer's cooperative ownership. A missing or inconsistent
context makes this opt-in comparison inconclusive. The returned context check
reports difference positions, but not the input IDs themselves.

This first contract is limited to the adapter's single eager DistilBERT
masked-LM encoder forward. It does not validate T5, decoder events, generation
or arbitrary hooks. It leaves the underlying event's generation fields unknown.
Legacy calls without the opt-in arguments retain their previous behavior.
Contexts retain full input IDs and masks and may contain sensitive data.

### Generation-aware comparisons

```python
from looking_glass.trace_schema import TraceEvent
from looking_glass.transformer_localization import TensorObservation, compare_transformer_traces

# `metadata` must be a validated schema-v1 event describing the actual snapshot.
reference = [TensorObservation(TraceEvent.from_dict(item.metadata()), item.value)
             for item in reference_session.events]
candidate = [TensorObservation(TraceEvent.from_dict(item.metadata()), item.value)
             for item in changed_session.events]
report = compare_transformer_traces(
    reference, candidate,
    reference_complete=True, candidate_complete=True, pairing_verified=True,
    execution_order_verified=True,
)
```

The example assumes that the producer has explicitly and accurately recorded
the generation context. For example, a standalone forward with no input cache
and no returned cache can use `GenerationContext(stage="forward", use_cache=False)`
when those facts are independently verified by the protocol. The flag alone does
not prove that the model did not consume previously supplied key/value states.
**The current GPT-2 trace adapter emits `stage="unknown"`; its
events are inconclusive until a caller supplies independently verified context.**
Do not infer cache positions from local token indices, or apply a single static
context to a multi-forward session without verifying each forward. For cached
prefill or decode, this first comparator requires an explicit generation step,
cache position, cache lengths before and after, and `use_cache=True` for each
event; other cases are conservatively unknown.

`CaptureDispatcher` saves each selected sample with its batch axis removed.
Use `observation_from_dispatch_slice(slice, generation=verified_context)` to
convert a real `CapturedSlice` into a comparison observation. The adapter
validates the dispatcher's documented single-row batch/token layout and makes
an unsqueezed view, without copying the tensor. The report preserves the
producer's saved shape and records the view shape and
`restore_singleton_batch_axis` transform. The view shares the capture's
storage, so the caller must keep the snapshot unchanged while comparing it.
Without an independently supplied generation context, the adapter preserves
the producer's unknown context and comparison remains inconclusive. An explicit
context cannot contradict a generation step already recorded by the producer.
No axis removal is inferred from an arbitrary schema event's shape.

Set `pairing_verified=True` only after the experiment protocol verifies the
relationship between reference and candidate runs: model and code revisions,
matched input identities and ordering, tokenization, capture policy, numerical
settings, and cache/forward conditions, allowing only the intended controlled
changes. A matching sample ID or coordinate does not verify those conditions.
Sample IDs that are present must match; if IDs are absent the external pairing
attestation must establish correspondence. Completion flags attest that all
events emitted under the declared tap policy were retained successfully. These
flags default to `False` and do not imply exhaustive coverage of the model.
Set `execution_order_verified=True` only when the producer guarantees the
supplied event sequence represents actual execution order. Schema sequence IDs
alone do not provide this guarantee across producers. Without it, a first
observed execution interval and a no-difference conclusion are inconclusive.

Each matched comparison reports original coordinates and each side's saved
shape, dtype, and device. Equality applies only to retained snapshots: a
lower-precision saved value can hide differences in the original source tensor.

Matching includes forward index, module path and kind, layer, invocation,
output path, original shape/dtype/device, ordered batch and token selections and
their axes, sample ID when known, and explicitly supplied generation metadata.
The two streams must have distinct run IDs, and each stream must have one valid
run ID. This prevents two changed events from claiming the same schema event
identity. Job and request IDs and capture mode may differ. A tensor copied
from GPU to CPU is checked against the separately recorded `saved_device`; its
original device is still part of the coordinate. Saved shape must agree with
the declared axis selections. Unknown required coordinates, missing or
inconsistent saved metadata, unmatched events, and incomplete streams cannot
establish equality. The report lists unknown fields and follows the status and
tolerance rules described for original probe events above.

The first-observed interval describes only supplied observation boundaries.
Its last-equal boundary, when available, uses the same forward, sample ID,
batch and token selection, axes, and generation context as the first unequal observation. Forward
indices must be contiguous and ordered; valid partial collections may therefore
remain inconclusive. Even matching selection labels require external validation
of sample correspondence.
Invocation is an ordinal and can fail to represent the same semantic call after
an unobserved branch change. A computation may change and later return to an
equal activation between taps. These comparisons identify observations to test;
they do not establish causal origins or explain a behavior. Interventions on
held-out inputs and appropriate controls remain necessary.

Invocation continuity is checked per module within each forward. Multiple
distinct slices from one invocation are allowed; duplicate full coordinates
remain ambiguous. A valid partial collection can be inconclusive; the tool does
not synthesize missing events or claim complete coverage. It does not save
activation values or impose its own tensor-memory limit.
