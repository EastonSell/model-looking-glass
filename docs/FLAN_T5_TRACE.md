# Eager FLAN-T5 encoder/decoder traces

For a bounded donor-state patch with same-site and off-target controls, see the
[T5 intervention workflow](FLAN_T5_INTERVENTION_WORKFLOW.md).

`looking_glass.flan_t5_trace` captures selected token vectors during one
standard `T5ForConditionalGeneration` forward. The caller supplies an already
loaded model, encoder IDs and mask, decoder IDs and mask; the adapter does not
download a model or tokenizer. It composes the existing `ActivationPlan` with
the shared `TraceEvent` metadata schema.

```python
from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward

# model is an already-loaded, float32, eager T5ForConditionalGeneration in eval mode.
output, records = trace_t5_forward(
    model,
    input_ids=source_ids, attention_mask=source_mask,
    decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
    run_id="example",
    points=[T5TracePoint("encoder", 0, "block", 0, 1),
            T5TracePoint("decoder", 0, "feed_forward", 0, 0)],
)
for record in records:
    metadata = record.event.to_dict()
    selected_vector = record.value
```

`T5TracePoint` fields are stack, zero-based layer, boundary, batch index, token
index, and optional invocation (currently only zero). Encoder and decoder
positions are local to their respective supplied tensors, whose sequence lengths
may differ. Both masks must be explicit binary tensors; selecting a masked
position is rejected. Records are returned in model execution order, with
explicit module path, stack-qualified module kind, layer, invocation, source
shape, output selector, and selected batch/token coordinates. Values are
detached clones. Distinct points at the same boundary may select different
tokens; duplicate points are rejected.
The event does not retain the input IDs or masks: preserve them separately
when comparing traces. A module path identifies the selected registered module,
not independently verified call-site provenance.

| Boundary | Module path | Meaning of captured tensor |
| --- | --- | --- |
| `self_attention` | `{stack}.block.{layer}.layer.0` | Tuple element 0 from `T5LayerSelfAttention`, **after** its residual addition. |
| `feed_forward` | Encoder `.layer.1`, decoder `.layer.2` | `T5LayerFF` tensor, **after** its residual addition. |
| `block` | `{stack}.block.{layer}` | Tuple element 0 after the whole block. A decoder block has already incorporated its cross-attention. |

The adapter does not expose a separate cross-attention output or attention
weights. It invokes the root model once with explicit `use_cache=False`, so
each record says `stage="forward"` and `use_cache=false`; generation step,
cache position and cache lengths remain unknown (`null`). It does not implement
`generate`, cached decoding, absolute positions across calls, beam search, or
attribution of decoder effects to cross-attention. A record is a local tensor
observation, not a causal or semantic claim.

The supported model is an unmodified standard T5 conditional-generation model
with eager attention, eval mode, float32 parameters on one CPU or CUDA device,
autocast disabled, no gradient checkpointing and no pre-existing local or
process-global forward hooks.
The adapter uses inference mode and the cooperative model lease provided by
`ActivationPlan`; callers remain responsible for GPU ownership and lifetime.
Hook checks are preflight checks under a cooperative process assumption; they
do not prevent another thread from installing a hook during the forward.
The current checkpoint smoke and unit tests run on CPU, not CUDA.

Limits apply to the number of points, aggregate captured bytes, each captured
boundary tensor, total input tokens and the expected full logits tensor. They
do not bound the model's intermediate allocations, all host memory, or GPU
context memory. Failed captures do not return partial records. The model output
is returned as well as the trace, so it remains the caller's responsibility to
release it.

`tools.flan_t5_trace_cpu_smoke` verifies the separately pinned FLAN-T5-small
asset manifest, uses one already exposed public feasibility case, and runs two
uncached CPU forwards: an untraced reference and a traced comparison. With
`CUDA_VISIBLE_DEVICES` empty and `HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=1`, run
it as a module from the repository root with `--assets <pinned-asset-dir>` and
`--output <new-report.json>`. It checks block outputs against the untraced
hidden-state outputs and wrapper outputs against independent baseline captures;
no generation or model download is performed.
The [saved CPU smoke report](../results/flan-t5-trace-cpu-smoke-v1.json) records
the pinned revision, asset and source hashes, eight trace metadata records and
comparison results. It is a saved test record under a trusted runtime, not an
attestation of execution.
