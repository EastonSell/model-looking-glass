# Donor-state interventions on eager FLAN-T5

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
either stack. As in the [trace guide](FLAN_T5_TRACE.md), self-attention and
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
execution, autocast, CUDA workflow or pre-existing hooks are supported. If a
`KeyboardInterrupt` or `SystemExit` occurs during a forward, PyTorch may skip
completion hooks and leave the plan's hooks and lease installed. Discard the
model and worker instead of attempting to reuse them.

For a runnable offline CPU example, the pinned FLAN-T5-small assets used by
the repository's [trace smoke](FLAN_T5_TRACE.md) can be reused:

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
