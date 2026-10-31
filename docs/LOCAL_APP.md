# Local model workbench

Clone, install, open the settings page, then try the built-in tiny demo or select an existing local checkpoint. Query it and inspect the captured vectors. **No checkpoint is downloaded by the app, and no prompt or activation is saved until you explicitly choose Save.** Hooks instrument the loaded runtime; they do not edit checkpoint files.

## Install and start

From a trusted clone with Python 3.11 or newer:

```bash
git clone --filter=blob:none --depth=1 https://github.com/EastonSell/model-looking-glass.git
cd model-looking-glass
python install.py
```

The installer creates this checkout's `.venv`, installs PyTorch 2.15.0 and application dependencies from official PyTorch/PyPI sources, and opens a loopback browser page. On Linux/Windows the default is **CUDA 13.2** (`torch==2.15.0+cu132`, from the [official wheel index](https://download.pytorch.org/whl/cu132/torch/)); this requires a compatible NVIDIA GPU and driver. It does not install or modify drivers. For a CPU-only system use `python install.py --backend cpu` (the default on macOS). The CPU build was exercised here; the CUDA build and hardware path were not. CUDA-capable PyTorch by itself does not establish a working GPU. Installing a Python project executes its packaging code; inspect a clone before trusting it. It does not install system services, configure credentials, or fetch models. Use `python install.py --dry-run` to inspect its commands, or `--no-launch` to install without opening a browser. An existing `.venv` requires explicit `--reuse-venv`; it is never silently replaced.

Later, start with `.venv/bin/python -m looking_glass.app` (Windows: `.venv\Scripts\python.exe -m looking_glass.app`). An already prepared environment can instead install `python -m pip install -e ".[app]"` and run `model-looking-glass`. Choose the matching PyTorch build first using its [official installation guidance](https://pytorch.org/get-started/locally/). An existing CPU-only `.venv` can be explicitly changed with `python install.py --reuse-venv --backend cuda`; build-local version pins ensure pip does not silently retain the CPU build. This is a package change, not a driver installation.

Options include `--no-browser`, `--port`, `--models-dir /explicit/model/folder`, and `--export-dir /chosen/export/folder`. `--models-dir` sets an explicit browsing root and lists immediate checkpoint candidates. Without it, enter a local folder and choose Browse to set the root, or paste a checkpoint path directly. The browser stays inside that root, does not scan home automatically, and never loads weights while listing folders. A `config.json` marks only a candidate; Inspect/Load still validates compatibility. Large folders are bounded and show a truncation notice. The server binds only to `127.0.0.1`, chooses a free port by default, and prints its URL. Keep the terminal running. Stop with Ctrl+C; a long active forward may need to finish first. This is a trusted single-user local tool, not a public service.

### If installation stops

Before installing packages, the installer checks the **environment's own Python**, not just the Python running `install.py`: it must be 3.11+, belong to this checkout's isolated `.venv`, and have pip. System-site-package environments and redirected `.venv` roots are rejected; a normal `bin/python` symlink to the base interpreter is allowed. `--dry-run` remains read-only and does not execute the environment.

Dependency commands ignore ambient pip/Python redirection variables and pip configuration files, require a virtual environment, and use the explicit official indexes shown by `--dry-run`. This prevents an unrelated `PIP_TARGET`, prefix or extra-index setting from silently redirecting this installation. Your environment and configuration files are not modified. This is install-target hygiene, not a sandbox for trusted package/build code.

An unsuccessful dependency step identifies which stage stopped and prints a copyable, explicit resume command using the same CPU/CUDA backend. Installed packages may remain, but the app is not launched and success is not reported. Resume uses `--reuse-venv --no-launch`; start the app separately after installation succeeds. No automatic retry, backend switch, deletion or environment recreation occurs.

If environment creation or its interpreter check fails, the existing directory is preserved. An older Python, missing pip or a moved/broken environment needs repair or a fresh environment; blindly adding `--reuse-venv` does not fix it. Keep any files you need, then use a fresh trusted checkout with supported Python rather than deleting an unknown environment. If installation completed but browser/app startup failed, the installer distinguishes that from a package failure and prints the direct app-start command.

## First capture without a checkpoint

Choose **Load tiny demo** to construct a fixed, tiny GPT-2 and byte-level tokenizer entirely in CPU RAM. No weights, tokenizer files or temporary checkpoint are written or downloaded. It is **random and untrained**: its output is not a useful answer or evidence of learned language, semantic features, model quality or GPU performance. The demo always uses CPU, independently of your local-checkpoint device selection.

Loading it replaces the loaded model and retained capture/comparison after success, so explicitly save anything you want first. The demo starts with auto hooks and short token limits; memory/event allowances and your checkpoint path/device preferences are unchanged. A suggested prompt fills only an empty prompt field; choose **Run** yourself. Then select an event and source position to inspect full values/bits, or try a controlled activation edit. The ordinary resource limits, exact capture and export paths still apply. The tiny model has a 128-position context window; byte-level tokens make ordinary text longer in tokens than a trained tokenizer would.

The demo's provenance, seed/configuration and untrained warning stay attached to captures and comparisons, including saved/reopened exports. A saved capture is not a saved model. Loading a fresh demo creates a new live-model binding; old or reopened captures cannot authorize edits to it. To use your own model afterward, load a supported local folder normally. Nothing about this demo broadens checkpoint compatibility.

## Checkpoint compatibility

A local Hugging Face-format directory must include `config.json`, matching **safetensors** weights (single file or indexed shards), and local tokenizer files or a ViT image processor configuration. A fast tokenizer is required for text models. The app rejects missing/mismatched tensors, custom remote code, quantization and pickle `.bin`-only checkpoints rather than silently constructing a different head.

| Family / exact head | Devices | Query | Auto capture points |
|---|---|---|---|
| GPT-2 / `GPT2LMHeadModel` (including compatible DistilGPT2) | CPU, CUDA path | Greedy text continuation | Attention, MLP and block outputs in every layer |
| DistilBERT / `DistilBertForMaskedLM` | CPU only | Fill explicit `[MASK]` positions | Attention, FFN and block outputs in every layer |
| T5 / `T5ForConditionalGeneration` (including compatible FLAN-T5) | CPU, CUDA path | Greedy text-to-text response; use the checkpoint's task prefix | Self-attention, feed-forward and block outputs in both stacks |
| ViT / `ViTForImageClassification` | CPU only | Classify an uploaded image | Attention, intermediate and block outputs in every layer |

**This is not support for every open checkpoint.** Llama, Qwen, Gemma, arbitrary architectures/heads, GGUF, remote-code models, quantized weights, multi-GPU dispatch, FP16/BF16 execution, KV-cache generation and training are not implemented here. Weights are loaded for float32 evaluation on one selected device. Compatibility remains bounded by the existing eager adapters; an architecture name alone is not proof that a customized checkpoint will work.

## Device selection and GPU semantics

**Auto** prefers available CUDA for GPT-2/T5. It chooses CPU for unavailable CUDA or CPU-only DistilBERT/ViT adapters and reports why. An explicit `cuda:N` request never silently falls back; unavailable devices and unsupported model families are rejected. Settings show discovered devices and the loaded model's actual device. Device indices follow PyTorch visibility (including `CUDA_VISIBLE_DEVICES`), not necessarily physical card numbering.

The app uses **direct forward hooks**: it copies the boundary values that the model actually computed. It does not implement a rounding-based gate, reconstruct hidden values from rounding error, or use the experimental auxiliary-gradient reporting channel. GPU arithmetic can differ from CPU arithmetic even at float32 (kernels, reduction order, TF32 settings and subnormal behavior); exact offload/save means preserving the observed values, **not** promising CPU/GPU-identical outputs. The run records device and numeric settings rather than silently changing global precision settings. See the [gradient-channel counterexample](GRADIENT_DENORMAL_V1.md).

GPU captures are moved synchronously to CPU RAM without dtype reduction. GPT-2 offloads at each selected hook; T5's existing adapter first retains one forward's captures on the GPU, then the app offloads and releases them before the next forward. This is deliberately conservative and can be slower than asynchronous/pinned-memory pipelines. No lossy compression, sampling or dropped vectors is hidden in the transfer.

## Full capture and readable inspection

1. Browse an explicitly chosen local folder or paste a checkpoint path. Inspect the model, select Auto/CPU/CUDA and the GPU allowance, then load it. Choose **Auto: all supported points**, or explicitly select roles and layers.
2. Set input/output-token limits and capture byte/event allowances. Prompt truncation is rejected, never silently performed.
3. Enter a prompt or image and run. A successful trace is exposed only after every selected capture and adapter cleanup completes. The run records its own model label/settings, input IDs and forward-step coordinates, independently of later model selections.
4. Search across the entire capture by module path/kind, layer or forward index, then select an event. Use the paginated **source-position picker**, previous/next vector buttons, or exact flat vector index. The viewer shows the recorded forward, layer, batch and token position/ID; T5 encoder and decoder streams are distinct. ViT shows CLS or patch coordinates and the processed-image pixel box, not original-image pixels. The raw table includes float32 values and hexadecimal bits, preserving distinctions such as signed zero. Pagination is presentation only: all captured values remain in RAM.
5. Optionally switch to **grouped coordinates**. This displays contiguous coordinate means and ranges; it is an explicitly approximate display, **not semantic concepts or inferred reasoning**. It does not replace, compress, or discard the original tensors.

Navigation uses only the capture’s recorded shape, selected source positions and matching forward inputs. Inconsistent or missing provenance is labeled **unknown**, while raw vectors remain available. Older exports may show exact token IDs with labels unavailable. Tokenizer pieces and single-token decoded fragments are captured from the run’s tokenizer, not a later loaded model; they are not necessarily words or contextual text. These optional display labels are capped at 512 characters each and 256 KiB total in compact UTF-8 JSON, with truncation/omission reported. This never truncates token IDs or activation tensors. A selected output token from forward N first becomes an input position at N+1, if that later forward actually ran; the final generated token may therefore have no captured input activation. Internally consistent coordinates are not independent proof of an imported export’s provenance.

Auto means every supported **module boundary**, not every mathematical operation, neuron state, attention probability, or internal node. T5 cross-attention is not a separate capture role in the current adapter. GPT-2/T5 use explicit uncached generation forwards: later steps recompute the growing prefix, and repeated token vectors belong to distinct forwards. Attention/MLP/block changes are observations, not a proof of a unique causal mechanism or complete interpretability.

## Try a controlled activation intervention

With a **fresh live GPT-2 capture** and a known selected vector, use the intervention panel to **zero that vector**, **add to one feature**, or **set one feature**. These are temporary runtime operations, never checkpoint/weight edits. Imported captures, another model load, unknown coordinates, and other architectures are read-only for this workflow. Reloading the same path does not restore the private live-model binding; make a fresh capture. Stale selections from another browser tab are rejected.

The experiment replays exactly one recorded input prefix and its attention mask, not the whole generated continuation. It runs an unhooked baseline, a same-site null control, the treatment, and an unhooked restoration. The null/restoration full logits must match the baseline bitwise; the fresh selected source must match the original capture. Other positions at the selected boundary must remain unchanged, and single-feature edits must preserve the other feature bits. A failed control publishes no successful comparison. This establishes a bounded same-input intervention comparison, **not** a model explanation, backdoor detector, general repair, or comparison between diverged generations.

The original RAM capture stays intact. The separate panel shows baseline/treatment next-token choices, full-logit change metrics, explicit no-op results, and four full selected-vector witnesses (null before/after and treatment before/after), with exact value/bit pagination. It is **not a full-model post-intervention trace**; full logits are compared but only metrics/top candidates are retained in the report. Choose **Save experiment** to export this separate result; the original Save still saves the original capture. Reopening a comparison permits inspection, not another live intervention; expand its run details to read the saved controls and metrics.

Experiments currently allow a fixed prefix of at most 64 tokens and at most 16 MiB of full logits. Their conservative logical-memory plan must fit both a 128 MiB additional allowance and the server’s remaining trace budget after the original capture. Limits reject requests before forwarding; they are not peak-memory guarantees. CPU execution is tested on tiny local checkpoints. CUDA uses the existing allocator/synchronization safeguards but remains hardware-unverified here. No permission to install a driver, download a model, or edit a checkpoint is implied.

## Memory, failures and privacy

The default trace allowance is 128 MiB; the server has a 512 MiB ceiling, which can be lowered with `--max-trace-mib`. Byte/event estimates aggregate **all requested generation forwards**, including repeated prefix captures. A too-large requested capture is rejected before forwarding when it exceeds the calculated bound. Runtime limits still apply. Select fewer boundaries or shorten the request rather than expecting automatic thinning.

The model default preflight cap is 2 GiB of parameter storage. Neither that nor the trace allowance is a peak-RAM guarantee: loading copies, logits, temporary outputs, tokenizer/image processing, metadata and export/import copies cost extra memory. The separate GPU allowance defaults to 2 GiB. During CUDA load/run, the app temporarily applies the stricter of the existing CUDA caching-allocator fraction and this allowance divided by visible device memory, and restores the prior fraction afterward. This process-global allocator setting is intended for the dedicated, serialized app process, not a shared inference service. Free-memory/headroom checks happen before work; synchronized peak allocated/reserved values are recorded. T5's per-forward retained GPU captures are included in preflight planning. `--max-vram-mib` limits the selectable allowance; `--max-model-mib` independently limits model parameter storage.

The allocator ceiling only covers PyTorch's CUDA caching allocator, **not all driver/library/context memory or other processes**. Free memory is an observation, not a reservation. Estimates cannot bound every kernel workspace, and GPU/CPU allocation can still fail. The app does not call `empty_cache` as proof that live work was safely freed. There is no asynchronous cancellation or general resource-isolation guarantee.

**Starting a new query replaces the previous RAM capture**, including when the new query later fails. The page warns before replacement; it does not show a confirmation dialog. Save first if you want to retain a run. Loading another model also clears the current capture. **Unload model** releases the resident model without clearing the current capture, which remains inspectable/saveable. Clear releases the capture separately.

A failure publishes no partial trace as complete. If execution leaves model ownership uncertain, the app marks it quarantined and requires a process restart rather than pretending cleanup succeeded. Ordinary validation errors before forwarding can be corrected without a restart.

Save writes a **new**, named directory under the export root, containing `tensors.safetensors`, `manifest.json` and a completion status. Existing directories are never overwritten. Exports include text prompts/token IDs, answers, image-coordinate metadata and full captured activations; treat them as potentially sensitive. The default export directory is Git-ignored. The saved-run picker lists only immediate folders under the export root. It labels apparently complete entries **unverified** and incomplete entries unavailable; listing never reads the tensor payload or proves integrity. It is a bounded convenience for a trusted local filesystem, not protection against a hostile concurrent filesystem writer. You can also paste an export path. Open saved trace verifies file inventory, shapes, byte limits and the tensor digest, then loads original values into RAM. Checksums detect changed bytes relative to the manifest; they do not authenticate a model or establish execution provenance. Model paths are labels, not cryptographic weight attestations. The checkpoint and original uploaded image are not copied into the trace export; keep them separately if needed for replay.

The server uses a fresh session token, same-origin/Host checks and no remote assets or telemetry. These are browser boundary safeguards, not protection from hostile code already running as the same local user. Do not proxy it to the internet or share its session page.

## Verification

```bash
python -m pytest tests/test_app_runtime.py tests/test_app_cuda.py \
  tests/test_app_selftest.py tests/test_app_storage.py \
  tests/test_app_http.py tests/test_app_install.py tests/test_app_interventions.py \
  tests/test_app_intervention_http.py tests/test_transformer_trace.py \
  tests/test_distilbert_trace.py tests/test_flan_t5_trace.py tests/test_vit_trace.py \
  tests/test_probes.py tests/test_interventions.py tests/test_intervention_plan.py \
  tests/test_model_ownership.py -q
# Optional developer-only DOM/API contract check (Node.js; no browser install):
node tests/app_ui_contract.cjs
```

The application dependency set is verified here on Linux, Python 3.12.14 and CPU PyTorch 2.15.0; other operating systems/interpreter combinations were not runtime-tested. The core library’s older Python 3.10 declaration is separate from this app’s Python 3.11+ dependency requirement.

Tests use tiny locally generated checkpoints and synthetic tensors; they do not establish behavior for every pretrained model. The broader research suite and experimental opcode boundaries remain separate; see the [documentation guide](README.md).

The DOM contract check exercises mocked requests, not rendering. Real loopback HTTP and tiny local-model roundtrips were exercised separately. Chromium visual testing was blocked by this execution environment, so no browser-rendering or cross-platform visual validation is claimed.

### CUDA validation boundary

This executor has no visible NVIDIA device and a CPU-only PyTorch build. CUDA policy tests use explicit synthetic API responses; they are **not GPU execution results**. Hardware-gated tests in `tests/test_app_cuda.py` exercise tiny local GPT-2/T5 models, same-device boundary-bit/selected-token comparisons and CPU-offload/save checks when real CUDA is available. Run them in the CUDA environment, without hiding the device:

```bash
python -m pytest tests/test_app_cuda.py -q -ra
```

The focused app/adapter suite completed with **472 passed and 4 real-CUDA prerequisite skips** on the CPU executor; the UI DOM contract covered 133 mocked authenticated requests, including demo onboarding, source-position paging, controlled interventions, stale cross-tab reads/saves, catalog failures, unload and failed tensor reads. Real loopback HTTP tests also exercised all three intervention operations, separate witness export/reopen and original-capture preservation. Independent local four-family roundtrips preserved all 105 inspected vectors’ raw bits and navigation coordinates after unloading the model and reopening an export. An independently installed wheel also passed authenticated demo loading/query, token-coordinate and static-asset checks. Demo tests verify no checkpoint writes/downloads, RNG preservation, and synthetic provenance through capture, intervention and export/reopen. Installer checks include real interpreter/config inspection plus simulated install-stage failures; no fresh package download or GPU installation was performed for those checks. These are focused checks, not a rerun of the entire historical research suite.

A prerequisite skip is not a pass. No GPU speedup, VRAM measurement, rounding-gate fidelity, or pretrained-model quality has been established by this release's CPU-only validation.

### Run the standalone GPU check

In the installed app environment:

```bash
python -m looking_glass.app_selftest --device cuda:0 --max-vram-mib 1024
# The installed console entry point is equivalent:
model-looking-glass-gpu-check --device cuda:0 --max-vram-mib 1024
```

This is an explicit, bounded diagnostic, not a background test when the app starts. It creates tiny deterministic synthetic GPT-2/T5 checkpoints in a temporary directory, uses only local dependencies, and never reads a user checkpoint or downloads a model. It checks actual device placement, same-device baseline versus traced full-logit bits, selected boundary bits, CPU offload, temporary export/reopen, repeated runs, hook cleanup and allocator-policy restoration. A failure stops further device work when ownership is uncertain rather than reporting recovery as success.

Standard output contains a JSON report with overall and per-stage **passed**, **failed**, or **skipped** states. Exit codes are **0 = passed, 1 = failed, 2 = skipped/prerequisite unavailable**. An unavailable CUDA device/build is skipped, never a CPU substitute or GPU pass. Device memory allowance covers the scoped PyTorch allocator only, with the same external-memory limitations described above. Temporary diagnostic checkpoints/traces are not your model data; the command does not persist a report unless you explicitly redirect its output.

A passing check validates only these tiny synthetic workloads on that machine and runtime. It does not establish arbitrary-checkpoint compatibility, performance improvements, rounding-gate behavior, or hostile-code isolation. The larger hardware-gated pytest checks remain available separately.
