# Bounded eager ViT image traces

`looking_glass.vit_trace` captures selected token vectors in one eager,
evaluation-mode `ViTForImageClassification` CPU forward. It composes the
existing `ActivationPlan` and `TraceEvent` schema. The caller supplies an
already-loaded model and a preprocessed CPU float32 `[batch, channel, height,
width]` tensor; the adapter does not download weights, load images, choose a
processor or infer an original image's dimensions.

```python
from looking_glass.vit_trace import ViTTracePoint, patch_token_index, trace_vit_classifier

# model is an eager float32 ViTForImageClassification in eval mode.
# pixels has the model's configured channel count and image height/width.
patch = patch_token_index(9, 7, grid_rows=14, grid_columns=14)
result = trace_vit_classifier(model, pixels, run_id="image-example", points=[
    ViTTracePoint(layer=2, module="block", batch=0, token=0),  # CLS
    ViTTracePoint(layer=2, module="block", batch=0, token=patch),
])
print(result.image_metadata, result.output.logits)
for record in result.records:
    print(record.event.to_dict(), record.token_kind, record.patch_row,
          record.patch_column, record.processed_pixel_box_xyxy, record.value)
```

Point fields are zero-based layer, module, batch, token and optional invocation
(only zero is supported). Token 0 is CLS; patch `(row, column)` maps to
`1 + row * grid_columns + column`. Each patch record supplies its row, column
and exclusive `(x0, y0, x1, y1)` box in **processed input pixels**. Rectangular
fixed image and patch dimensions are supported when each divides evenly. The
CLS record has no patch box. The supplied pixel tensor's shape, patch size,
grid size, ordering and CLS index are returned separately. These coordinates
do not recover a crop/resize transform or a box in the original image: the
caller must preserve that preprocessing and image identity. Once attention has
mixed positions, a token's state may include information from other patches.

| Module | Captured tensor | Width |
| --- | --- | --- |
| `attention` | Projected self-attention branch before the first residual addition; not attention probabilities. | `hidden_size` |
| `intermediate` | Activated first MLP projection before its output projection and second residual. | `intermediate_size` |
| `block` | Complete block output after the second residual addition. | `hidden_size` |

Records use detached selected vectors and identify the exact path, layer,
invocation, local batch/token, full output shape and output selector. They are
returned in model execution order (layer, then attention/intermediate/block,
then batch/token), independent of the order supplied. The run ID is limited to
128 characters. The model output is the
ordinary image-classifier logit object, not generated text. Generation/cache
fields remain unknown; the generic transformer comparator has no ViT image
pairing contract. The [ViT trace comparison helper](#comparing-paired-vit-traces)
checks explicitly paired processed-grid coordinates. A coordinate is an
observation, not a causal localization or evidence of a particular visual concept.

The adapter requires a standard, unmodified eager ViT classifier and patch
embedding geometry, exact configured input dimensions, eval mode, no gradient
checkpointing, CPU float32 parameters and buffers on CPU, no autocast or
compiled execution, and no existing local or global forward hooks. It disables
attention/hidden-state collection and positional interpolation for its single
forward. The boundary meanings above were verified with the project's pinned
Transformers 4.57.3. Preflight does not authenticate checkpoint or library
bytes or arbitrary changes to internal forwards; callers must establish those
assumptions. Caller-owned exclusive model access is required across the call;
the process-local lease is cooperative and cannot stop foreign hooks or another
process. If `KeyboardInterrupt` or `SystemExit` interrupts a forward, PyTorch
may skip completion hooks and leave a lease/hook installed: discard the model
and worker instead of reusing them.

Limits cover input pixel bytes, each selected full module tensor, event count,
aggregate retained vector bytes and output logits. They do not bound weights,
temporary allocations, metadata, peak memory, or memory outside this process.
No partial result is returned on failure. Some returned values are inference
tensors; clone them outside inference mode before editing. The output and
vectors can be sensitive; choose carefully where they are saved or shared.

An offline example requires the project transformer extra and Pillow (the
recorded example used Pillow 11.3.0), verifies the three pinned
`WinKawaks/vit-tiny-patch16-224` assets and accepts a local image. The optional flag below verifies one image
already exposed in the repository's v1 study; that image is not redistributed.
The script uses the cached slow `ViTImageProcessor`, converts to RGB without
EXIF transpose, and records original dimensions and hashes in its report.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -B -m examples.vit_trace_cpu --assets <pinned-asset-dir> --image <local-image> --output <new-report.json> --verify-public-case
```

The example compares trace values and logits with a separate direct-hook CPU
forward and saves a report with hashes and metadata, not the raw image, logits
or activation vectors. Omit `--verify-public-case` to use another local image;
its hash will be included in the report. The check demonstrates mechanics on
that input and runtime, not model accuracy, device performance or a visual
explanation. The [sample report](../examples/vit-trace-cpu-sample.json) was
generated with Pillow 11.3.0 on the exposed v1 image; the image itself is not
included and numeric hashes can vary across runtimes.

## Comparing paired ViT traces

`compare_vit_traces` compares selected CLS and processed-grid patch vectors from two successful `trace_vit_classifier` calls. It uses the existing exact-value trace comparison for equality and returns descriptive distances for each aligned vector. It does not run the model, load images or infer why an activation changed.

```python
from looking_glass.vit_trace import ViTTracePoint, trace_vit_classifier
from looking_glass.vit_trace_comparison import compare_vit_traces

# Same owned, evaluation-mode CPU ViT; two already-preprocessed pixel tensors.
points = [ViTTracePoint(0, "block", 0, 0),   # CLS
          ViTTracePoint(0, "block", 0, 1)]   # first processed-grid patch
reference = trace_vit_classifier(model, reference_pixels, points=points, run_id="reference")
candidate = trace_vit_classifier(model, candidate_pixels, points=points, run_id="candidate")

# Set this only after checking model/runtime, preprocessing, intended image
# pairing, and complete execution of the same selected policy yourself.
report = compare_vit_traces(reference, candidate, points=points, pairing_verified=True)
print(report["status"], report["comparisons"])
```

The declared `points` must be nonempty and distinct. The helper sorts them in the adapter's execution order and requires both results to contain exactly that policy, with distinct run IDs and matching full source and saved shapes. It checks each record's layer, module path and kind, invocation, batch/token coordinate, selector, dtype/device and sequence order. It also verifies the reported image, patch and grid dimensions, the CLS index, row-major patch token mapping and each processed-pixel box. Both results must use identical processed geometry and the same batch coordinates. A mismatched or incomplete result, a nonfinite vector, an exceeded byte limit, or the default `pairing_verified=False` produces `status="inconclusive"` with reasons and no partial comparison rows. Malformed API arguments raise an exception.

`pairing_verified=True` is a **caller attestation**. The trace records do not establish that the same weights and runtime were used, that either trace came from the supplied pixels, that preprocessing was equivalent, or that the selected image pair is scientifically appropriate. Matching grid indices and processed-pixel boxes identify tensor coordinates; they do not establish matching objects, semantic regions, original-image pixels or causal influence. Preserve source images and preprocessing records separately when those facts matter.

Each row reports the coordinate, exact numerical equality, counts of numerically and bitwise changed float32 elements, L2 distance, maximum absolute difference and mean absolute difference. Numerical equality treats `+0` and `-0` as equal; the bitwise count distinguishes them. Finite float32 bit patterns are decoded before binary64 arithmetic so subnormal differences are retained even when the host flushes float32 denormals. Distances are descriptive binary64 calculations, not exact real-arithmetic norms, significance tests or localization intervals. Results depend on the chosen inputs and selected boundaries; there is no claim of exhaustive model coverage.

`max_events` defaults to 128 (maximum 1024), and `max_trace_bytes` defaults to 8 MiB (maximum 64 MiB) for the combined retained vectors in the two supplied results. The helper copies one pair of selected vectors at a time. These limits do not bound model weights, input images, logits, Python objects, temporary allocator usage or total process memory. Callers must keep the results and their tensor storage quiescent during comparison. The returned report contains coordinates and run IDs, not the raw activation vectors or image pixels.

## Bounded ViT donor-state interventions

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

The trace boundaries described above apply here: `attention` is
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
