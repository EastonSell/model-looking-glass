# Local model workbench

Clone, install, open the settings page, select an existing local checkpoint, then query it and inspect the captured vectors. **No checkpoint is downloaded by the app, and no prompt or activation is saved until you explicitly choose Save.** Hooks instrument the loaded runtime; they do not edit checkpoint files.

## Install and start

From a trusted clone with Python 3.11 or newer:

```bash
git clone --filter=blob:none --depth=1 https://github.com/EastonSell/model-looking-glass.git
cd model-looking-glass
python install.py
```

The installer creates this checkout's `.venv`, installs the tested CPU PyTorch 2.15.0 and application dependencies from the official PyTorch/PyPI package sources, and opens a loopback browser page. Installing a Python project executes its packaging code; inspect a clone before trusting it. It does not install system services, configure credentials, or fetch models. Use `python install.py --dry-run` to inspect its commands, or `--no-launch` to install without opening a browser. An existing `.venv` requires explicit `--reuse-venv`; it is never silently replaced.

Later, start with `.venv/bin/python -m looking_glass.app` (Windows: `.venv\Scripts\python.exe -m looking_glass.app`). An already prepared environment can instead install `python -m pip install -e ".[app]"` and run `model-looking-glass`. Choose a CPU PyTorch build first if needed; the generic package index can supply a larger accelerator build.

Options include `--no-browser`, `--port`, `--models-dir /explicit/model/folder`, and `--export-dir /chosen/export/folder`. `--models-dir` lists only immediate checkpoint subdirectories; otherwise enter an explicit checkpoint directory in settings. The server binds only to `127.0.0.1`, chooses a free port by default, and prints its URL. Keep the terminal running. Stop with Ctrl+C; a long active forward may need to finish first. This is a trusted single-user local tool, not a public service.

## Checkpoint compatibility

A local Hugging Face-format directory must include `config.json`, matching **safetensors** weights (single file or indexed shards), and local tokenizer files or a ViT image processor configuration. A fast tokenizer is required for text models. The app rejects missing/mismatched tensors, custom remote code, quantization and pickle `.bin`-only checkpoints rather than silently constructing a different head.

| Family / exact head | Query | Auto capture points |
|---|---|---|
| GPT-2 / `GPT2LMHeadModel` (including compatible DistilGPT2) | Greedy text continuation | Attention, MLP and block outputs in every layer |
| DistilBERT / `DistilBertForMaskedLM` | Fill explicit `[MASK]` positions | Attention, FFN and block outputs in every layer |
| T5 / `T5ForConditionalGeneration` (including compatible FLAN-T5) | Greedy text-to-text response; use the checkpoint's task prefix | Self-attention, feed-forward and block outputs in both stacks |
| ViT / `ViTForImageClassification` | Classify an uploaded image | Attention, intermediate and block outputs in every layer |

**This is not support for every open checkpoint.** Llama, Qwen, Gemma, arbitrary architectures/heads, GGUF, remote-code models, quantized weights, GPU execution, KV-cache generation and training are not implemented here. Weights are loaded for CPU float32 evaluation. Compatibility remains bounded by the existing eager adapters; an architecture name alone is not proof that a customized checkpoint will work.

## Full capture and readable inspection

1. Inspect/load the local model. Choose **Auto: all supported points**, or explicitly select roles and layers.
2. Set input/output-token limits and capture byte/event allowances. Prompt truncation is rejected, never silently performed.
3. Enter a prompt or image and run. A successful trace is exposed only after every selected capture and adapter cleanup completes. The run records its own model label/settings, input IDs and forward-step coordinates, independently of later model selections.
4. Select an event, then a vector/token and coordinate page. The raw table includes float32 values and hexadecimal bits, preserving distinctions such as signed zero. Pagination is presentation only: all captured values remain in RAM.
5. Optionally switch to **grouped coordinates**. This displays contiguous coordinate means and ranges; it is an explicitly approximate display, **not semantic concepts or inferred reasoning**. It does not replace, compress, or discard the original tensors.

Auto means every supported **module boundary**, not every mathematical operation, neuron state, attention probability, or internal node. T5 cross-attention is not a separate capture role in the current adapter. GPT-2/T5 use explicit uncached generation forwards: later steps recompute the growing prefix, and repeated token vectors belong to distinct forwards. Attention/MLP/block changes are observations, not a proof of a unique causal mechanism or complete interpretability.

## Memory, failures and privacy

The default trace allowance is 128 MiB; the server has a 512 MiB ceiling, which can be lowered with `--max-trace-mib`. Byte/event estimates aggregate **all requested generation forwards**, including repeated prefix captures. A too-large requested capture is rejected before forwarding when it exceeds the calculated bound. Runtime limits still apply. Select fewer boundaries or shorten the request rather than expecting automatic thinning.

The model default preflight cap is 2 GiB of parameter storage. Neither that nor the trace allowance is a peak-RAM guarantee: loading copies, logits, temporary outputs, tokenizer/image processing, metadata and export/import copies cost extra memory. CPU-only does not mean allocation cannot fail. There is no asynchronous cancellation or general resource-isolation guarantee.

A failure publishes no partial trace as complete. If execution leaves model ownership uncertain, the app marks it quarantined and requires a process restart rather than pretending cleanup succeeded. Ordinary validation errors before forwarding can be corrected without a restart.

Save writes a **new**, named directory under the export root, containing `tensors.safetensors`, `manifest.json` and a completion status. Existing directories are never overwritten. Exports include text prompts/token IDs, answers, image-coordinate metadata and full captured activations; treat them as potentially sensitive. The default export directory is Git-ignored. Open saved trace verifies file inventory, shapes, byte limits and the tensor digest, then loads original values into RAM. Checksums detect changed bytes relative to the manifest; they do not authenticate a model or establish execution provenance. Model paths are labels, not cryptographic weight attestations. The checkpoint and original uploaded image are not copied into the trace export; keep them separately if needed for replay.

The server uses a fresh session token, same-origin/Host checks and no remote assets or telemetry. These are browser boundary safeguards, not protection from hostile code already running as the same local user. Do not proxy it to the internet or share its session page.

## Verification

```bash
python -m pytest tests/test_app_runtime.py tests/test_app_storage.py \
  tests/test_app_http.py tests/test_app_install.py -q
# Optional developer-only DOM/API contract check (Node.js; no browser install):
node tests/app_ui_contract.cjs
```

The application dependency set is verified here on Linux, Python 3.12.14 and CPU PyTorch 2.15.0; other operating systems/interpreter combinations were not runtime-tested. The core library’s older Python 3.10 declaration is separate from this app’s Python 3.11+ dependency requirement.

Tests use tiny locally generated checkpoints and synthetic tensors; they do not establish behavior for every pretrained model. The broader research suite and experimental opcode boundaries remain separate; see the [documentation guide](README.md).

The DOM contract check exercises mocked requests, not rendering. Real loopback HTTP and tiny local-model roundtrips were exercised separately. Chromium visual testing was blocked by this execution environment, so no browser-rendering or cross-platform visual validation is claimed.
