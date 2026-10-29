# GPT-2 head-slice observation and patching

`run_gpt2_head_slice` observes or replaces one attention-head slice entering a GPT-2 attention output projection. It composes the copying study's existing `HeadProjectionWitness` with the shared cooperative model lease. It runs one eager, evaluation-mode, uncached CPU `GPT2LMHeadModel` forward on one unpadded token row. It neither loads nor authenticates a checkpoint; the caller owns the model and token identities.

## Boundary and coordinates

For the tested Transformers 4.57.3 implementation, each attention layer concatenates the individual head outputs and passes `[batch, local_token, hidden]` to `transformer.h.{layer}.attn.c_proj`. Head `h` occupies `[h * head_width, (h + 1) * head_width)` along its last axis. A `GPT2HeadSlicePoint` selects a zero-based layer, local token and head; batch and invocation are explicitly zero. The API checks an unpruned standard eager attention/projection layout at entry and at the projection input, rejects a missing or repeated projection invocation, and requires all input mask entries active. It does not support padding, cached decoding, generation, cross-attention, altered implementations, pruning, training, autocast, compilation, or multiple simultaneous owners.

The recorded slice is **before** the projection mixes heads. It is not an attention probability, a record of which earlier positions supplied the values, a post-projection head contribution, or proof of a unique circuit. Equal input L2 changes at different heads do not imply equal projected or downstream effects. Selected coordinates are local to this explicit forward; token meaning and alignment across calls are the caller's responsibility.

## Use

```python
import torch
from looking_glass.gpt2_head_slice import GPT2HeadSlicePoint, run_gpt2_head_slice

# Supply an already-loaded eager CPU float32 GPT2LMHeadModel in eval mode.
ids = torch.tensor([[15496, 995]], dtype=torch.int64)
point = GPT2HeadSlicePoint(layer=0, token=1, head=0)
observed = run_gpt2_head_slice(model, ids, point=point, run_id="observe")

# A same-site self patch is a no-change control; separate calls are sequential.
same = run_gpt2_head_slice(model, ids, point=point, run_id="self",
                           replacement=observed.head_before.clone())
zeroed = run_gpt2_head_slice(model, ids, point=point, run_id="zero",
                             replacement=torch.zeros_like(observed.head_before))
print(zeroed.metadata.to_dict())
```

`metadata` identifies the module **input** and its positional argument, input shape, input IDs and all-one attention mask, head interval, dtype/device, layer, token, batch, invocation and uncached call. It is deliberately a separate input-boundary record: the existing `TraceEvent` describes module outputs and must not be relabeled as this input. `before`, `after`, and `projection_output` retain the complete projection-input before and after tensors and the complete projection output, respectively; `head_before` and `head_after` are views into the first two. The supplied replacement is copied before the call. The witness checks the requested slice bit-for-bit, every untouched input bit, and the input actually received by the projection in the same forward. A successful return includes the resulting logits and means the one declared projection invocation completed; a failed call returns no partial result. Detached tensors created during inference may need cloning outside inference mode before editing.

`max_input_bytes`, `max_tensor_bytes`, `max_capture_bytes`, `max_source_bytes` and `max_logits_bytes` are checked before the forward. `tensor_bytes` counts the three retained full projection witnesses; logits and the replacement have their own limits. Bounds exclude model parameters, Python metadata, allocator overhead, temporary copies and peak process memory. Inputs and witnesses may contain sensitive information and are not persisted automatically.

The cooperative lease and hooks require exclusive, quiescent model ownership. They do not stop foreign code, another process or native work, authenticate model weights, or establish semantic pairing. Ordinary Python forward failures remove hooks and release the lease when cleanup is verified. If hook cleanup is uncertain, or a cancellation-style `BaseException` interrupts the call, the model is quarantined: discard the model or worker rather than trying to reuse it. No CUDA execution is supported.

## Offline example

With the pinned DistilGPT2 files already in a Hugging Face hub cache, run:

```text
python examples/gpt2_head_slice_cpu.py --cache-dir /path/to/hub --output head-slice.json
```

The example uses the previously public `Hello world` token IDs `[15496, 995]`, verifies a clean/observed/self-patch/clean-repeat control, then separately zeros head 0 and head 1 at the last token of layer 0. It reports each realized input displacement and observed logit change; the second head is a descriptive comparison, not a null or dose-matched control. The JSON is a compact explanation rather than a complete saved-tensor replay. [A sample](../examples/gpt2-head-slice-cpu-sample.json) is provided; numeric values can vary with runtime and hardware. These artificial interventions demonstrate API behavior, not an explanation of ordinary model behavior.
