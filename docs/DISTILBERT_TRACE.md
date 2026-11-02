# DistilBERT masked-LM trace adapter

`trace_distilbert_masked_lm` performs one eager, evaluation-mode CPU forward
of an already-loaded `DistilBertForMaskedLM` and captures selected outputs
using the existing ordered activation plan. Each observation carries a
`trace_schema.TraceEvent` plus a detached tensor snapshot. The adapter does
not load or authenticate weights and does not perform generation.

For an artificial injection, same-site restoration and off-target control,
see [Test a known DistilBERT intervention](#test-a-known-distilbert-intervention).

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
[transformer comparisons](LOCALIZATION.md#transformer-observations) for the full contract.
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

## Test a known DistilBERT intervention

`run_known_distilbert_intervention` composes the existing bounded DistilBERT
trace and ordered activation plan. It adds a chosen scalar to one hidden
coordinate at an early block output, then tests a clean-state replacement at
the **same token and same later block** in the injected run. A separate
forward replaces the later-block state at a different, active token with that
token's clean state. This is an artificial mechanics experiment, not a search
for a natural circuit or proof of a unique location.

```python
from looking_glass.distilbert_intervention_workflow import run_known_distilbert_intervention

# model is an eager, evaluation-mode CPU float32 DistilBertForMaskedLM.
# ids and mask are a single aligned CPU int64 [1, length] row.
result = run_known_distilbert_intervention(
    model, ids, mask, injection_layer=0, restoration_layer=1,
    target_token=2, control_token=5, readout_position=6,
    readout_token_id=2003, coordinate=0, amount=5.0,
)
print(result.report["restoration"])
print(result.report["off_target"])
print(result.report["readout"]["measurements"])
```

The caller supplies the aligned token IDs, complete binary attention mask,
early and later layer indices, target and control positions, hidden coordinate,
and masked-LM readout position and vocabulary ID. The later layer must follow
the earlier one; target, control and readout positions must be active in the
mask, and target and control must differ. The workflow accepts 2–16 input
tokens, hidden width at most 1,024, and vocabulary at most 65,536. It does not
infer which position is a mask or whether the supplied tokens are meaningful.
DistilBERT is bidirectional and the readout is a logit at an **input** position,
not a generated next-token prediction. Other padded positions may be present;
their exact mask values are recorded.

### What is measured

The clean donor trace records both selected positions at the early and later
block outputs. Each intervention uses its own sequential `ActivationPlan`.
The workflow retains whole early-block outputs before and after injection,
and whole later-block outputs immediately before and after a patch. It checks
the same injection lands in every run, only the requested hidden coordinate
changes, both patch receivers match the injected baseline, each selected output
equals its corresponding clean donor, and every other output position remains
bit-identical. The self-patch uses the later receiver's own same-forward value
and must reproduce injected logits. An immediate injection-site undo restores
the captured pre-injection state and must reproduce clean logits; it is a
by-construction implementation check. Clean and injected repeats must match
their respective full logits. Failures do not publish a result.

The report contains the complete input IDs and mask, plan action and trace
metadata, selected raw logits, distances from the clean and injected **readout
vectors**, and actual input-state L2 displacements at injection, restoration
and the off-target patch. The off-target patch is a real comparison only when
its reported status is `active`: a zero displacement or zero injection is
labeled unavailable. The patches need not have equal doses, equal downstream
sensitivity, or equal causal roles. A zero injection or target displacement, an
unchanged chosen readout, or an unavailable control makes `restoration_comparison` unavailable;
all measured values remain in the report. Even when that status is
`descriptive`, it makes no pass/fail or generalization claim.
Scalar readouts, vector distances and L2 doses are calculated in binary64
from the recorded float32 bit patterns, preserving float32 subnormals even
when CPU arithmetic flushes them.

`result.witnesses` retains the clean selected vectors (with their documented
singleton axes), every plan's full captured outputs, and all six readout-logit
vectors. `result.report` is JSON-compatible and includes trace schemas and
ordered plan metadata; it is a summary rather than a complete execution proof.
Both returned dictionaries and tensors are mutable. They may contain
sensitive inputs or activations; callers must decide how to protect, save and
audit them. The trace and plan caps are reported, and the returned tensor
payload is bounded by `18 * full_output_bytes + 6 * hidden_vector_bytes +
6 * vocabulary_size * 4`. These bounds do not cover model weights, full MLM
logits, Python metadata, temporary allocations, or peak memory.

The model must be a standard eager `DistilBertForMaskedLM` in evaluation mode,
on CPU in float32, with no existing local/global hooks and no autocast or
compiled execution. The existing trace and plan leases protect each session;
the caller must prevent noncooperating or interleaved model use and model
mutation **across** all nine forwards. Input clones and dimensions are checked
before and after each call, but this cannot detect a mutation that is reversed
within a call. No training gradients, generation or concurrent model use are
supported. An uncertain fatal interruption can leave a plan and its model
quarantined; discard the model or worker when cleanup cannot be established,
as explained in [ordered plans](INTERVENTION_PLANS.md). The checkpoints,
tokenizer and semantic alignment are not authenticated by this workflow.

### Offline pretrained example

From the checkout with PyTorch and `transformers==4.57.3` installed, use a
cache containing the public pinned checkpoint:

```sh
python -m examples.distilbert_intervention_workflow \
  --cache-dir /path/to/huggingface/hub --output distilbert-intervention.json
```

Add `--witnesses /path/to/new-witnesses.pt` to save the tensor witnesses. The
output paths must be new. The example requests a cached revision with
`local_files_only=True`; it does not independently verify the cached asset
bytes. Its input is the already-public agreement sentence used by the
[trace example](#running-the-cached-example). A [sample summary](../examples/distilbert_intervention_workflow.sample-summary.json)
omits the detailed event/plan metadata and tensors; the actual JSON report
includes the metadata. Numeric values can vary across runtimes. In that sample,
both the later target and off-target replacements delivered nonzero, unequal
doses; they are not matched controls. Do not load untrusted PyTorch files.
