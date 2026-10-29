# DistilBERT masked-LM trace adapter

`trace_distilbert_masked_lm` performs one eager, evaluation-mode CPU forward
of an already-loaded `DistilBertForMaskedLM` and captures selected outputs
using the existing ordered activation plan. Each observation carries a
`trace_schema.TraceEvent` plus a detached tensor snapshot. The adapter does
not load or authenticate weights and does not perform generation.

```python
from looking_glass.distilbert_trace import trace_distilbert_masked_lm

# model is an evaluation-mode CPU float32 DistilBertForMaskedLM with eager
# attention; ids and mask are matching CPU int64 [batch, token] tensors.
result = trace_distilbert_masked_lm(
    model, ids, mask, run_id="example", layers=(0, 2),
    modules=("attention", "ffn", "block"), samples=(0,), tokens=(6,),
    max_events=6, max_trace_bytes=32768, max_tensor_bytes=16 * 1024 * 1024,
)
print(result.output.logits.shape)
for observation in result.events:
    print(observation.event.to_dict(), observation.value.shape)
```

The adapter always passes `output_attentions=False`,
`output_hidden_states=False` and `return_dict=True`. This matters because in
the supported eager Transformers implementation a block's tuple starts with
attention weights when attentions are requested, rather than its hidden
output. The adapter does not support attention-weight capture.

| Role | Module path for layer `N` | Captured output |
| --- | --- | --- |
| `attention` | `distilbert.transformer.layer.N.attention` | Tuple element 0: projected contextual attention output, before the attention residual and layer normalization. This is not attention weights. |
| `ffn` | `distilbert.transformer.layer.N.ffn` | Tensor output of the feed-forward network, before the second residual and layer normalization. |
| `block` | `distilbert.transformer.layer.N` | Tuple element 0: the final hidden state after the block's residuals and layer normalizations. |

The model is a bidirectional encoder. MLM logits are indexed by **input
position**, normally a masked position; they are not next-token generation
logits. A capture at a padded query position can still have a contextual
value. The result retains the exact supplied input IDs and binary attention mask
in an immutable `encoder_context`, and also exposes the mask as tuples. Token
and sample coordinates are local indices for this forward.

There is one event per selected layer, role, sample and token, in actual
layer/attention/FFN/block completion order, with invocation and forward index
zero. Selected layers are sorted and module roles follow that execution order,
regardless of their input-list order; sample and token selectors retain the
caller-supplied order. The plan originally captures a `[hidden]` vector. The observation's
`value` is a no-copy `[1, 1, hidden]` view that restores the selected batch and
token axes; `original_saved_shape` and `comparison_transform` disclose that
view. The returned tensors are mutable and remain owned by the caller. Event
metadata includes the source and saved shapes, dtype, device, output selector
and positions. It does not claim a unique internal mechanism.

The shared event schema's generation/cache fields remain `unknown`/`None`;
the adapter does not invent cache state. To compare two complete results from
this adapter, opt in to its separate encoder-forward contract:

```python
from looking_glass.transformer_localization import compare_transformer_traces

# reference and candidate are results of separate adapter calls with distinct
# run IDs. The caller has independently checked the model, runtime and tap
# policy, and knows these inputs have identical IDs and masks.
report = compare_transformer_traces(
    reference.events, candidate.events,
    encoder_contexts=(reference.encoder_context, candidate.encoder_context),
    expected_input_differences=(),
    reference_complete=True, candidate_complete=True,
    pairing_verified=True, execution_order_verified=True,
)
```

For intentional input changes, supply the **exact** tuple of changed
`(batch_index, token_index)` positions instead of `()`. The comparator checks
both full masks, input shapes and IDs, the declared changes, and the complete
selected layer/module/sample/token event policy. Matching local positions do
not prove semantic or tokenizer alignment. Context and model provenance are
producer/caller claims, not execution or weight-integrity attestations; see
[transformer comparisons](TRANSFORMER_LOCALIZATION.md) for the full contract.
Without it, unknown generation metadata remains inconclusive.

## Running the cached example

With PyTorch and `transformers==4.57.3` installed, run from an installed
checkout using a cache that already contains the pinned public checkpoint:

```sh
python -m pip install -e ".[transformer]"
python examples/distilbert_trace_example.py --cache-dir /path/to/huggingface/hub --output distilbert-trace.json
```

The example uses `local_files_only=True`, the pinned
`distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, and an already-public agreement
row from the [CPU evaluation](../results/bert-agreement-cpu-evaluation-v1-20261029/README.md).
It compares clean and traced logits and saves metadata and four-value previews,
not full logits or snapshots. The [sample summary](../examples/distilbert_trace_example.sample.json)
is illustrative and may vary by runtime. The adapter itself does not verify
cache hashes, and the example requests a cached revision without independently
verifying its bytes; use a trusted, independently verified checkpoint when
needed. For the recorded sample, the operator separately checked the cached
weight file against the previously published SHA-256 pin
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`;
that check did not authenticate the other cached files.

## Limits and ownership

The adapter supports only standard eager `DistilBertForMaskedLM` modules on
CPU in float32, in evaluation mode, with explicit binary int64 attention
mask, no CPU autocast and no preexisting local or global forward hooks. Selected module aliases
are rejected. It supports no gradients, cached generation, concurrent or nested
model use, compiled execution, distributed execution, or cross-process model
ownership. `ActivationPlan` uses a cooperative model lease: the caller must
prevent noncooperating threads and model-tree or weight mutation. Cleanup and
uncertain-interruption limitations are described in [ordered plans](INTERVENTION_PLANS.md).

`max_events` limits retained observations, `max_trace_bytes` limits the
aggregate selected snapshots, and `max_tensor_bytes` limits each full module
output. Checks fail rather than silently truncate. These bounds do not include
model weights, MLM logits, input-ID/mask tuples, metadata, temporary tensors, allocator overhead or
peak memory. The result is published only after a successful forward and plan
cleanup. Input clones and recorded dimensions are checked before and after the
call; this cooperative check cannot detect a mutation that is later reversed.
Traces and contexts can contain sensitive data and stay in the process unless
a caller saves them.
