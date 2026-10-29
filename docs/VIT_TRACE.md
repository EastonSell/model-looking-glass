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
pairing contract. A coordinate is an observation, not a causal localization
or evidence of a particular visual concept.

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
