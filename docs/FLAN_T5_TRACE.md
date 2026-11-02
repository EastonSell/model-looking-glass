# Eager FLAN-T5 encoder/decoder traces

This guide also covers [paired forward comparisons](#paired-uncached-t5-forward-comparisons)
and [donor-state interventions](#donor-state-interventions-on-eager-flan-t5).

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

For an explicit two-stack input/mask comparison on CPU, use the paired
uncached-forward contract below.

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

`synchronous_cpu=True` explicitly opts a CPU trace into the
[synchronous CPU plan wrapper](INTERVENTION_PLANS.md). The caller must
ensure exclusive, quiescent ownership and synchronous CPU execution with no
deferred native or device work; tensor-device checks do not establish these
conditions. A supported root-call exception can unwind the plan's Python hooks
and cooperative lease without returning a partial trace. An uncertain plan
transition, lifecycle race or known non-CPU output instead requires retaining
the model, error/traceback and accessible plan and resource roots and
quarantining the worker. The default remains
a direct forward, and CUDA lifetime and cleanup are not established by this
option.

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

## Paired uncached T5 forward comparisons

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

The comparator conservatively treats any `t5_` module kind or module path
beginning `encoder.block.` or `decoder.block.` as requiring this T5 context.
Other producers that use those path prefixes will therefore be inconclusive
unless they satisfy this narrow contract; existing BERT and GPT-2 paths do not
use those prefixes.

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
comparison. The boundary semantics above also apply here.

### Offline CPU smoke

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

## Donor-state interventions on eager FLAN-T5

`compare_t5_donor_patch` composes the existing T5 trace adapter and
`ActivationPlan` to capture two runs and patch a selected state. The caller
supplies an already-loaded, standard eager `T5ForConditionalGeneration` in
evaluation mode, float32 on a little-endian CPU. It performs six separate forwards without
cache or generation. This example tests what a specific intervention does; it
does not locate a unique mechanism or establish a natural representation.

```python
from looking_glass.flan_t5_trace import T5TracePoint
from looking_glass.flan_t5_intervention_workflow import compare_t5_donor_patch

# All six inputs are single-row CPU int64 tensors. Encoder and decoder lengths
# may differ; each shared mask has the same shape as its two corresponding IDs.
result = compare_t5_donor_patch(
    model,
    receiver_input_ids=receiver_source, donor_input_ids=donor_source,
    attention_mask=source_mask,
    receiver_decoder_input_ids=receiver_decoder, donor_decoder_input_ids=donor_decoder,
    decoder_attention_mask=decoder_mask,
    target=T5TracePoint("encoder", 0, "block", 0, 6),
    control=T5TracePoint("encoder", 0, "block", 0, 5),
    readout_position=1, readout_token_id=15,
)
print(result.report["realized_l2"], result.report["readout"])
target_before = result.witnesses["plans"]["target"]["before"]
target_after = result.witnesses["plans"]["target"]["after"]
```

The two selected points must be distinct active tokens at the **same** stack,
layer and boundary, batch zero and invocation zero. Each token index is local
to its encoder or decoder tensor; matching numbers across stacks do not imply
aligned positions. The receiver and donor must have identical shapes and share
their exact masks within each stack. Both ID sequences are retained in the
report, so differences in source inputs or teacher-forced decoder prefixes are
visible. The caller must establish whether positions and inputs have the
intended semantic alignment; matching shapes does not establish it.

The workflow traces both tokens in a receiver and donor forward, applies a
same-site self patch to the receiver, patches the receiver's target with the
donor's target, patches the distinct control token with its own donor value,
and repeats the clean receiver. Each arm retains the **full** selected module
output immediately before and after the action. It checks the patch against its
donor bits, checks every other position is unchanged, and checks self-patch and
repeat results against the receiver. An off-target patch is a comparison, not a
null: it may affect the readout, have a different L2 dose, or have zero dose.
Zero displacement is reported as unavailable for that comparison. The target
can also be zero. These checks do not make the two interventions dose-matched
or prove model, runtime or semantic identity between calls.

Supported boundary names are `block`, `self_attention`, and `feed_forward` in
either stack. As described above, self-attention and
feed-forward wrapper outputs include their residuals; a decoder block output
includes cross-attention. There is no separate cross-attention tap. The readout
is a vocabulary logit vector at the specified **active decoder input position**
of a teacher-forced forward; it is not a generated token. If donor and receiver
decoder prefixes differ, their readout logits have different conditioning and
should not be treated as directly paired without further justification.

The result contains a JSON-compatible report and detached CPU tensor witnesses.
Some witnesses are inference-mode tensors; clone them outside inference mode
before editing their values.
It records exact IDs and masks, trace events, plan action metadata, tensor shapes,
input-state L2 displacements and readout summaries. It uses retained binary32
bits to calculate distances, including subnormals. Limits bound selected module
outputs, aggregate returned witness tensors, input tokens and each full logits
tensor; they do not bound model weights, temporary activations or peak memory.
The current numeric decoder also limits each hidden or vocabulary vector to
1 MiB. A failed workflow raises without returning partial results.
The caller must exclusively own the model throughout all six calls, including
against noncooperating threads or other processes. Hooks are sequential and
the underlying lease is cooperative. No `generate`, cached decoding, compiled
execution, autocast, CUDA workflow or pre-existing hooks are supported. By
default the model is called directly. If a `KeyboardInterrupt` or `SystemExit`
occurs during a direct forward, PyTorch may skip completion hooks and leave the
plan's hooks and lease installed; discard the model and worker.

For an exclusively owned, quiescent model whose operations are synchronously
completed on CPU, a caller may explicitly pass `synchronous_cpu=True`. All six
forwards then use the [synchronous CPU plan wrapper](INTERVENTION_PLANS.md),
including receiver, donor and repeat traces. When an exception propagates from
the wrapper's own root model call under its stated contract, it can invalidate
the failed evidence and remove its hooks and cooperative lease after Python
unwinds; no partial workflow result is returned. The caller still must verify
model and runtime state before reuse. Known non-CPU outputs, reentrant or
foreign lifecycle activity, or an interrupted/uncertain plan transition can
leave hooks and the lease retained. Retain the model, error and traceback,
including accessible plan and other resource roots, and quarantine the worker
when cleanup or ownership is uncertain. Visible CPU
tensor checks do not prove there was no hidden device work, deferred native
operation, or noncooperating thread; that is the caller's responsibility. This
option makes no CUDA cleanup or device-quiescence claim.

For a runnable offline CPU example, the pinned FLAN-T5-small assets used by
the trace smoke above can be reused:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -B -m examples.flan_t5_donor_patch_cpu --assets <pinned-asset-dir> --output <new-report.json>
```

The example verifies the pinned asset files, uses an already-public cat/dog
feasibility pair, and checks encoder and decoder block patches. Its JSON report
contains summaries and metadata, not the raw in-memory witnesses; retain those
separately if an independent replay is needed. Values can differ across runtime
and CPU environments, and the example is a mechanics check, not a result about
translation. A [sample CPU report](../examples/flan-t5-donor-patch-cpu-sample.json)
records one such run; it is a summary rather than independent execution proof.
