# Comparing transformer observations

`looking_glass.transformer_localization` compares tensor payloads attached to
`trace_schema.TraceEvent` metadata. It uses the existing conservative
localization comparator after encoding explicit source coordinates. It does not
automatically accept or infer metadata from an arbitrary model hook.

## DistilBERT encoder-forward comparisons

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

## Generation-aware comparisons

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
tolerance rules in [LOCALIZATION.md](LOCALIZATION.md).

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
