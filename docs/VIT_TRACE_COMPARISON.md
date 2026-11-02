# Comparing paired ViT traces

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
