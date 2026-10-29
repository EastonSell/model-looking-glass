# Paired uncached T5 forward comparisons

The generic transformer comparator cannot establish that two T5 forwards used
paired encoder **and** decoder inputs from their trace events alone. This
separate CPU-only contract records both ID tensors, both masks, model
dimensions, and the complete selected T5 tap policy for each forward.
The underlying eager T5 tracer remains unchanged.

```python
from looking_glass.flan_t5_trace import T5TracePoint
from looking_glass.t5_forward import trace_t5_comparable_forward
from looking_glass.transformer_localization import compare_transformer_traces

points = (T5TracePoint("encoder", 0, "block", 0, 1),
          T5TracePoint("decoder", 0, "block", 0, 1))
# Both calls supply explicit CPU int64 encoder and decoder IDs/masks to the
# same separately checked CPU float32, eager, eval T5 model.
left = trace_t5_comparable_forward(model, input_ids=left_ids, attention_mask=left_mask,
    decoder_input_ids=left_decoder, decoder_attention_mask=left_decoder_mask,
    points=points, run_id="left")
right = trace_t5_comparable_forward(model, input_ids=right_ids, attention_mask=right_mask,
    decoder_input_ids=right_decoder, decoder_attention_mask=right_decoder_mask,
    points=points, run_id="right")
report = compare_transformer_traces(left.events, right.events,
    t5_contexts=(left.t5_context, right.t5_context),
    expected_t5_encoder_differences=((0, 1),),
    expected_t5_decoder_differences=(),
    reference_complete=True, candidate_complete=True,
    pairing_verified=True, execution_order_verified=True)
```

The example declares exactly one changed encoder ID and no changed decoder ID.
Use explicit tuples of zero-based `(batch, token)` coordinates for **both**
stacks; `()` declares no changes, while an omitted declaration is not known.
The full encoder masks must match each other, as must the full decoder masks;
masked changed or selected tokens are unsupported. Paired shapes must match
within each stack, but the encoder and decoder sequence lengths need not match
each other. Every event must match the full ordered policy, source shape,
stack-qualified path and module kind, selected batch/token, tuple selector,
forward index and invocation. A missing context, mismatched policy, unsupported
cache metadata or undeclared change leaves the comparison **inconclusive**.
A previous equal encoder event is not used as the preceding boundary for a
decoder difference. Within a stack, an observed interval concerns only the
selected points and is not proof of a causal origin.

The producer records immutable copies of the IDs and masks but retains caller-owned
mutable tensor values. It first clones the supplied CPU int64 input tensors and
checks their device, dtype, shape and values and the model dimensions after the forward; transient mutations
that are later reversed cannot be detected. Original `[hidden]` captures are
no-copy views shaped `[1, 1, hidden]` for the comparator; the original saved
shape and transformation are recorded on each observation. The contract caps
input IDs at 4,096 tokens and selected points at 128, and inherits the tracer's
capture, tensor and logits limits. These limits do not bound model weights,
all intermediate allocations, allocator overhead or total host memory.

Input equality and the declared changed coordinates do not establish semantic
or tokenizer alignment, model or weight identity, runtime equivalence, or the
actual execution order. The caller's four boolean attestations remain required;
the context is producer-supplied evidence, not an execution attestation. The
model and process must remain cooperative during tracing. Contexts contain the
input IDs and masks and can be sensitive. The wrapper calls one uncached
forward per trace with caller-supplied decoder IDs (`use_cache=False`); cache
positions, lengths and generation step remain unknown. There is no support or
claim for `generate`, cached decoding, cross-attention isolation or GPU
comparison. See [T5 tracing](FLAN_T5_TRACE.md) for boundary semantics.

## Offline CPU smoke

`tools.flan_t5_paired_compare_cpu_smoke` verifies the pinned FLAN-T5-small
assets and uses a previously exposed public feasibility pair. From the
repository root, with the checkpoint already present:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B -m tools.flan_t5_paired_compare_cpu_smoke \
  --assets /path/to/pinned-assets --output /path/to/new-report.json
```

The [saved smoke report](../results/flan-t5-paired-compare-cpu-smoke-v1.json)
records three CPU forwards: identical inputs compare equal, the predeclared
source-token change produces a difference, and omitting that declaration is
inconclusive. It records source and asset pins, event-value hashes and
comparator output. It is a limited reproducibility check, not an experiment
about translation, causal localization, performance or generation.
