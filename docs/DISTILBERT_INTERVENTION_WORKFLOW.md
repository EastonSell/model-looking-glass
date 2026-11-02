# Test a known DistilBERT intervention

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

## What is measured

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

## Offline pretrained example

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
[trace example](DISTILBERT_TRACE.md). A [sample summary](../examples/distilbert_intervention_workflow.sample-summary.json)
omits the detailed event/plan metadata and tensors; the actual JSON report
includes the metadata. Numeric values can vary across runtimes. In that sample,
both the later target and off-target replacements delivered nonzero, unequal
doses; they are not matched controls. Do not load untrusted PyTorch files.
