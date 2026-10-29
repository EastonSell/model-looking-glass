# Trace and intervene on a pretrained GPT-2 model

This small CPU example uses the repository's existing trace and ordered
intervention APIs directly. It runs five forwards of the pinned DistilGPT2
model on the public input `Hello world`: a clean reference, a bounded trace,
a same-site self-patch control, a deliberately chosen activation edit, and a
final clean check after the sessions close. It saves a short JSON summary.

## Run it

Install PyTorch and the optional transformer dependencies in a Python 3.10+
environment, from the repository root:

```sh
python -m pip install -e ".[transformer]"
python examples/pretrained_trace_intervention.py \
  --cache-dir /path/to/huggingface/hub --output gpt2-trace-example.json
```

Replace the cache path with a writable Hugging Face hub cache holding the
pinned files. The loader is offline by default and verifies the assets before
loading them. If the files are not already cached, add `--allow-download` to
explicitly fetch them. The weights are about 353 MB and loading temporarily
stages an additional copy inside the cache. See the
[pretrained loader guide](PRETRAINED_MODEL.md) for the exact revision, asset
hashes, dependencies and cache behavior. The output filename must be new.

## What to inspect

The script encodes the fixed prompt as token IDs `[15496, 995]`, then records
both positions at three boundaries in layer 0. `attn` is the attention module
output after its projection, **not attention weights**; `mlp` is the MLP
output before its residual addition; `block` is the block output after its
residual additions. For this model each captured tensor has shape `[1, 2,
768]`. The trace is capped at three events and 32 KiB of captured tensor
values; the captured values occupy 18,432 bytes. This limit does not bound the
model, logits, intermediate tensors, or peak memory. The JSON saves metadata
and a four-coordinate preview of the last token, not the full trace. In the
script, `event.value` is the full selected snapshot while the process runs.

The intervention selects batch 0, token 1 at `transformer.h.0.mlp`. It
captures the vector, adds `5.0` to coordinate 0, then captures the resulting
vector in the same forward. A separate self-patch plan captures the vector
and writes that same vector back. The summary reports exact comparisons of
the clean, traced, self-patched and post-session logits, and verifies that the
edited vector equals the captured vector plus the specified delta. It also
checks that the first-token logits are unchanged, the final-token logits
change, and both sets of logits are finite. These
checks are for this evaluation-mode CPU run; exact floating-point values can
vary across environments. Self-patching is an inference control, not a
gradient-preserving no-op.

The [saved sample](../examples/pretrained_trace_intervention.sample.json)
was produced with the cached pinned checkpoint, Python 3.11, PyTorch 2.8.0 and
Transformers 4.57.3 on CPU. The observed displacement had L2 norm 5 and the
largest absolute logit change was about 0.601. The most likely next-token ID
was 13 (`.`) in both the clean and edited passes. That entry is an argmax of
the last-position logits, not a generated sentence.

This is a known artificial edit and an observation of its effect on one
input. It does not identify a natural feature, explain the model's ordinary
behavior, or establish a general causal mechanism. Trace and intervention
sessions require exclusive model ownership and are deliberately sequential.
See [GPT-2 tracing](TRANSFORMER_TRACE.md) and
[ordered plans](INTERVENTION_PLANS.md) for the full API and failure contracts.
