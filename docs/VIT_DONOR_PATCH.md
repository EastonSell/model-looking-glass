# Bounded ViT donor-state interventions

`compare_vit_donor_patch` compares interventions in one caller-owned,
evaluation-mode eager `ViTForImageClassification` on CPU. It takes two already
processed float32 images of the same configured shape and explicit target and
control `ViTTracePoint`s. The points must select distinct tokens in batch zero at
the same layer, boundary and invocation. Either can be CLS or a patch. Token 0
is CLS; use `patch_token_index` for the row-major **processed** patch grid.
Those coordinates do not establish an original-image crop, a shared object or
semantic correspondence between the two images.

```python
from looking_glass.vit_intervention_workflow import compare_vit_donor_patch
from looking_glass.vit_trace import ViTTracePoint, patch_token_index

# model is an eager, CPU float32 ViT image classifier in eval mode; each input
# is a single, already processed CPU float32 [1, channels, height, width] tensor.
patch = patch_token_index(9, 7, grid_rows=14, grid_columns=14)
result = compare_vit_donor_patch(
    model, receiver_pixels=receiver, donor_pixels=donor,
    target=ViTTracePoint(2, "block", 0, 0),
    control=ViTTracePoint(2, "block", 0, patch), readout_class=4)
print(result.report["sites"], result.report["doses"], result.report["readout"])
```

The workflow captures both states on the receiver and donor. It then runs a
same-site self patch, a no-op, the target donor patch, and an off-target donor
patch, followed by a receiver repeat. When the target displacement is nonzero,
it also attempts a same-site reflected control: the float32-rounded vector
`receiver - (donor - receiver)`. It does not run that arm if its ideal value
is outside the finite float32 range, its rounded value is nonfinite, or its
realized displacement is zero. The report calls an active
control `matched` only when its **realized** L2 displacement is within the
supplied relative tolerance of the target dose; it otherwise says `unmatched`,
`zero`, or gives an unavailable reason. The off-target arm is descriptive and
may have a different or zero dose. A reflected direction is an active
comparison, not an established semantic null or a matched causal control.

Each intervention saves the full selected module output immediately before
and after the operation. It checks the selected source's exact float32 bits at
the requested site and unchanged bits at every other site. Self/no-op logits,
receiver-repeat logits and selected receiver values must match bit for bit;
all intervention pre-states must agree. The report includes action metadata,
image geometry, selected class-logit changes, control results, input hashes,
and measured doses. `result.witnesses` contains detached receiver/donor
vectors, full before/after tensors and logits; some are inference-mode tensors,
so clone them outside inference mode before modifying them. These checks
establish mechanics for the supplied model and pair. They do not establish
model identity, meaningful alignment, a visual concept, generalization, or
causality beyond the specified forward interventions.

Boundaries have the meanings in [VIT_TRACE.md](VIT_TRACE.md): `attention` is
the projected branch before the first residual, `intermediate` is the activated
MLP projection, and `block` is the full block output. Each selected vector is
at most 1 MiB for the binary32 numeric decoder. Positive limits bound the
aggregate input bytes, each full selected output, logits and all **returned
tensor witnesses**. The reported retained bound excludes model weights, input
copies, metadata, temporary/native allocations, error graphs and peak memory.
No partial successful result is returned on a failed check. Hashes and
activations may disclose information about inputs; handle the report and
witnesses accordingly.

The caller must provide exclusive model access, stable model weights and
runtime, and defensible input pairing. The lease is cooperative: it does not
prevent unrelated threads, processes, or untracked mutations. The workflow
uses the same direct forward path and preconditions as the ViT tracer. If
`KeyboardInterrupt` or `SystemExit` interrupts a forward, PyTorch can leave
hooks and the lease installed; discard the model and worker rather than reuse
them. Ordinary exceptions propagate and may be reused only if cleanup
actually completed. This CPU workflow does not provide CUDA/native-work
synchronization or recovery.

For a small offline mechanics example using only random weights and synthetic
pixels, install the project transformer extra and run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -B -m examples.vit_donor_patch_cpu
```

It prints a bounded summary and does not load an image, download weights, or
claim a result about a trained model.
