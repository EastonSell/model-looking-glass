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

Choose the folder that actually contains these files, not a model's online repository ID. In an existing Hugging Face cache, that is usually `models--…/snapshots/<revision>/`, not the cache parent or `blobs/` folder. Keep all referenced weight shards together. A common text export contains `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and any tokenizer-specific special-token files. Compatible legacy GPT-2 vocabulary/merges and DistilBERT `vocab.txt` layouts can also construct a fast tokenizer; a tokenizer JSON is not the only accepted layout.

**Inspect folder** now reports a separate setup checklist without replacing the loaded model: configuration and weight headers checked, tokenizer/processor locally loadable, **weights not loaded and inference/device execution not tested**. Text token IDs must fit the model's vocabulary (padded model vocabularies are allowed); DistilBERT needs its mask token and T5 its decoder-start ID. ViT requires matching bounded RGB processor/model dimensions. These structural checks cannot prove that arbitrary tokenizer vocabulary meanings match particular weights; use the assets from the same checkpoint. Inspection is not a cryptographic fingerprint, and files can change afterward.

Input assets are bounded before parsing: at most 64 MiB per large tokenizer asset, 1 MiB for small configuration/special-token/template files, and 128 MiB combined. These are file-size limits, not peak-memory guarantees. Missing assets, unsupported conversion dependencies and malformed input configuration produce setup guidance rather than automatic downloads or package installation. From an existing trusted environment, export the **matching** model with `model.save_pretrained(folder, safe_serialization=True)` and its tokenizer with `tokenizer.save_pretrained(folder)` (or its ViT processor with `processor.save_pretrained(folder)`); the app does not perform this export for you.

Load repeats input preparation before retiring the current owner or loading new weights, then reuses that prepared tokenizer/processor. A failed input preflight leaves the old model, private live binding and retained capture intact. The UI also keeps the capture readable on later load failure, while refreshing the actual model state: a weight-allocation/transfer failure can still retire or quarantine the model, so further live interventions may require a fresh load and capture. Only a successful replacement clears the previous capture/comparison.

| Family / exact head | Devices | Query | Auto capture points |
|---|---|---|---|
| GPT-2 / `GPT2LMHeadModel` (including compatible DistilGPT2) | CPU, CUDA path | Greedy text continuation | Attention, MLP and block outputs in every layer |
| Llama / standard `LlamaForCausalLM`, default or static Llama 3 RoPE | CPU, CUDA path | Greedy plain-text continuation; no automatic chat template | Full attention, MLP and decoder-block outputs in every layer |
| DistilBERT / `DistilBertForMaskedLM` | CPU, CUDA path | Fill explicit `[MASK]` positions | Attention, FFN and block outputs in every layer |
| T5 / `T5ForConditionalGeneration` (including compatible FLAN-T5) | CPU, CUDA path | Greedy text-to-text response; use the checkpoint's task prefix | Self-attention, feed-forward and block outputs in both stacks |
| ViT / `ViTForImageClassification` | CPU, CUDA path | Classify an uploaded image | Attention, intermediate and block outputs in every layer |

**This is not support for every open checkpoint.** Qwen, Gemma, Mistral, multimodal/Llama-derived architectures with different implementations, arbitrary heads, GGUF, remote-code models, quantized weights, multi-GPU dispatch, FP16/BF16 execution, KV-cache generation and training are not implemented here. Weights are loaded for float32 evaluation on one selected device. Compatibility remains bounded by the existing eager adapters; an architecture name alone is not proof that a customized checkpoint will work.

Llama support is deliberately bounded to the installed Transformers 4.57.3 standard eager `LlamaForCausalLM` with `model_type="llama"` and `pretraining_tp=1`. Ordinary RoPE (`rope_scaling` absent or null) and the exact static `rope_type="llama3"` frequency transformation are supported. Standard multi-head and grouped-query attention are checked against actual projection dimensions. Linear, dynamic, YaRN, LongRoPE, partial rotary dimensions and custom implementations remain rejected rather than silently changed. Use matching local fast-tokenizer assets (`LlamaTokenizerFast`, standard GPT-2 BPE or compatible fast JSON assets); no SentencePiece conversion dependencies, weights or chat templates are downloaded. The prompt is supplied as raw text: an instruction-tuned checkpoint may need its own correctly formatted prompt. The app does **not** apply its chat template automatically, and only fresh privately bound captures enable the fixed-prefix intervention panel described below.

For static Llama 3 RoPE, the configuration must contain exactly `rope_type`, `factor`, `low_freq_factor`, `high_freq_factor` and `original_max_position_embeddings`. Factors must be finite (`factor >= 1`, `0 < low_freq_factor < high_freq_factor`); the original context length must be a positive integer below `max_position_embeddings`. The adapter checks the standard initializer and unchanged rotary-buffer bits across each forward. It uses the [pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_rope_utils.py#L453-L515), not a replacement approximation. A large configured context is not a promise it will fit: uncached prefix work, full logits and every selected vector still count against the app limits.

Float32 needs four bytes per parameter before captures, logits and loading copies: a billion parameters alone is about 3.7 GiB, beyond the default 2 GiB model ceiling. For a larger compatible checkpoint, set `--max-model-mib` explicitly and, on CUDA, `--max-vram-mib` plus the Settings allowance only when your machine has sufficient headroom. Increasing these limits does not create memory or guarantee a successful load.

## Verified pretrained example: SmolLM2-135M (CPU)

[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2) is one actual pretrained checkpoint verified here, pinned to **`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`**. Its publisher declares Apache-2.0 and public, ungated access. The weight file is **269,060,552 bytes**; all downloaded files together are about 260 MiB. The app loads its BF16-stored weights into **CPU float32** (538,060,032 parameter bytes), using the matching GPT-2 BPE tokenizer with the standard Llama model. This pairing is supported without renaming or editing checkpoint assets.

On two short prompts, complete logits and all **90 attention/MLP/block outputs** matched independent same-device controls bit-for-bit. An installed-wheel HTTP run of “Gravity is” generated four tokens and retained **360 events / 1,620 full vectors** (3,732,480 tensor bytes). Every vector's values, bits and coordinates survived explicit Save, Unload, Clear and Open. Nothing was exported before Save; source files, tied weights and rotary buffers remained unchanged. See the [exact record and asset hashes](../results/local-app-smollm2-135m-cpu-20261031/record.json).

To reproduce, explicitly obtain that revision's files into a local folder; this is an optional download, not something app startup performs. With the app environment active:

```bash
hf download HuggingFaceTB/SmolLM2-135M README.md config.json generation_config.json merges.txt model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --local-dir ./models/smollm2-135m
python -m looking_glass.app --models-dir ./models
```

Select the checkpoint folder and CPU, then query a short prompt. The original transparency scripts and record are preserved unchanged as historical evidence. To replay both against their original behavior, use implementation commit `0d1dc22ed82c2c4b965f754eba8409cea996bcc1` (for example, in a separate checkout) as `--implementation-root`; the old workbench script explicitly expects that version's then-read-only Llama workflow:

```bash
python results/local-app-smollm2-135m-cpu-20261031/verify_capture.py --checkpoint ./models/smollm2-135m --implementation-root /path/to/original-checkout
python results/local-app-smollm2-135m-cpu-20261031/verify_workbench.py --checkpoint ./models/smollm2-135m --implementation-root /path/to/original-checkout --export-root ./local-verification-export
```

The export directory must be new. These are short CPU transparency/round-trip checks—not GPU validation, a quality benchmark, full-context testing or proof that every Llama checkpoint works. Wall times in the record are observations, not performance claims. No checkpoint weights or raw activation payloads are included in the repository.

## Device selection and GPU semantics

**Auto** prefers available CUDA for all five supported families and chooses CPU when CUDA is unavailable, reporting why. An explicit `cuda:N` request never silently falls back; unavailable devices and unsupported model families are rejected. Settings show discovered devices and the loaded model's actual device. Device indices follow PyTorch visibility (including `CUDA_VISIBLE_DEVICES`), not necessarily physical card numbering.

The app uses **direct forward hooks**: it copies the boundary values that the model actually computed. It does not implement a rounding-based gate, reconstruct hidden values from rounding error, or use the experimental auxiliary-gradient reporting channel. GPU arithmetic can differ from CPU arithmetic even at float32 (kernels, reduction order, TF32 settings and subnormal behavior); exact offload/save means preserving the observed values, **not** promising CPU/GPU-identical outputs. The run records device and numeric settings rather than silently changing global precision settings. See the [gradient-channel counterexample](GRADIENT_DENORMAL_V1.md).

GPU captures are moved synchronously to CPU RAM without dtype reduction. GPT-2 offloads at each selected hook; T5, DistilBERT, ViT and Llama first retain one forward's complete selected captures on the GPU, then the app offloads and releases them. DistilBERT predicts all supplied mask positions in one forward, not autoregressive generation. This is deliberately conservative and can be slower than asynchronous/pinned-memory pipelines. No lossy compression, sampling or dropped vectors is hidden in the transfer.

The standalone DistilBERT adapter remains CPU-only by default. Its explicit `synchronous_cpu=False` opt-in permits one CUDA float32 device, validates inputs, logits and captures on that device, and synchronizes before returning. CUDA results deliberately have no CPU `EncoderForwardContext`: the historical CPU comparison contract is not asserted for a GPU run. Callers must still own the model and handle failed/uncertain CUDA execution; the app quarantines such failures.

ViT image decoding and preprocessing remain on CPU, with bounded RGB output geometry checked before transfer. Its CUDA adapter hashes the exact processed pixel encodings through blocking integer-bit CPU copies before/after forwarding; it does not infer original-image coordinates or change pixel precision. Like DistilBERT, standalone `trace_vit_classifier` is CPU-only by default and accepts CUDA only with `synchronous_cpu=False`. Its historical CPU-only comparison API remains unchanged and rejects CUDA-origin events, even after their vectors are offloaded.

## Full capture and readable inspection

1. Browse an explicitly chosen local folder or paste a checkpoint path. Inspect the model, select Auto/CPU/CUDA and the GPU allowance, then load it. Choose **Auto: all supported points**, or explicitly select roles and layers.
2. Set input/output-token limits and capture byte/event allowances. Prompt truncation is rejected, never silently performed.
3. Enter a prompt or image and run. A successful trace is exposed only after every selected capture and adapter cleanup completes. The run records its own model label/settings, input IDs and forward-step coordinates, independently of later model selections.
4. Search across the entire capture by module path/kind or layer, or filter by exact **source position**, **forward** and **stack**. The Source column labels verified single-vector events; multi-vector events show a count rather than pretending to represent one token. Selecting a filtered match opens its actual vector, even when it is not vector 0. Positions are zero-based: T5 decoder position 0 is the start token; ViT position 0 is CLS, followed by row-major processed-image patches. Module search does not search tokenizer text. Clear filters to reveal unknown-source events, which cannot match exact source filters. Use the paginated **source-position picker**, previous/next vector buttons, or exact flat vector index. The viewer shows the recorded forward, layer, batch and token position/ID; T5 encoder and decoder streams are distinct. ViT shows CLS or patch coordinates and the processed-image pixel box, not original-image pixels. The raw table includes float32 values and hexadecimal bits, preserving distinctions such as signed zero. Pagination is presentation only: all captured values remain in RAM.
5. Optionally switch to **grouped coordinates**. This displays contiguous coordinate means and ranges; it is an explicitly approximate display, **not semantic concepts or inferred reasoning**. It does not replace, compress, or discard the original tensors.

Navigation uses only the capture’s recorded shape, selected source positions and matching forward inputs. Inconsistent or missing provenance is labeled **unknown**, while raw vectors remain available. Older exports may show exact token IDs with labels unavailable. Tokenizer pieces and single-token decoded fragments are captured from the run’s tokenizer, not a later loaded model; they are not necessarily words or contextual text. These optional display labels are capped at 512 characters each and 256 KiB total in compact UTF-8 JSON, with truncation/omission reported. This never truncates token IDs or activation tensors. A selected output token from forward N first becomes an input position at N+1, if that later forward actually ran; the final generated token may therefore have no captured input activation. Internally consistent coordinates are not independent proof of an imported export’s provenance.

Browsing reuses shared forward-input validation **only within one read request**; every event still undergoes its own source/shape/identity checks, and nothing is cached across reads or saved in exports. On this Linux/Python 3.12 CPU executor, a paired synthetic-catalog benchmark (six layers, 512 tokens, width 768; seven alternating measurements after warmup) reduced a 9,216-event DistilBERT stack filter from **414 ms to 73 ms median**. A GPT-2 100-vector picker page fell from 9.1 ms to 5.3 ms. Complete responses and tensor-byte hashes matched; ViT saw little change. These are storage-method timings, not browser rendering, inference or GPU speedups, and are not performance guarantees.

Auto means every supported **module boundary**, not every mathematical operation, neuron state, attention probability, or internal node. T5 cross-attention is not a separate capture role in the current adapter. GPT-2/T5/Llama use explicit uncached generation forwards: later steps recompute the growing prefix, and repeated token vectors belong to distinct forwards. Attention/MLP/block changes are observations, not a proof of a unique causal mechanism or complete interpretability.

## Try a controlled activation intervention

With a **fresh live GPT-2 or supported standard Llama capture** and a known selected vector, use the intervention panel to **zero that vector**, **add to one feature**, or **set one feature**. These are temporary runtime operations, never checkpoint/weight edits. Imported captures, another model load, unknown coordinates, and other architectures are read-only for this workflow. Reloading the same path does not restore the private live-model binding; make a fresh capture. Stale selections from another browser tab are rejected.

The experiment replays exactly one recorded input prefix and its attention mask, not the whole generated continuation. It runs an unhooked baseline, a same-site null control, the treatment, and an unhooked restoration. The null/restoration full logits must match the baseline bitwise; the fresh selected source must match the original capture. Other positions at the selected boundary must remain unchanged, and single-feature edits must preserve the other feature bits. Llama additionally checks static rotary-buffer/configuration identity and bits, unchanged exact input IDs/mask, full same-device logits and absence of a KV cache on every trial. A failed control publishes no successful comparison. This establishes a bounded same-input intervention comparison, **not** a model explanation, backdoor detector, general repair, or comparison between diverged generations.

The original RAM capture stays intact. The separate panel shows baseline/treatment next-token choices, full-logit change metrics, explicit no-op results, and four full selected-vector witnesses (null before/after and treatment before/after), with exact value/bit pagination. It is **not a full-model post-intervention trace**; full logits are compared but only metrics/top candidates are retained in the report. Choose **Save experiment** to export this separate result; the original Save still saves the original capture. Reopening a comparison permits inspection, not another live intervention; expand its run details to read the saved controls and metrics.

Experiments currently allow a fixed prefix of at most 64 tokens and at most 16 MiB of full logits. Their conservative logical-memory plan must fit both a 128 MiB additional allowance and the server’s remaining trace budget after the original capture. Limits reject requests before forwarding; they are not peak-memory guarantees. CPU execution is tested on tiny local checkpoints and the pinned pretrained SmolLM2 example below. CUDA uses the existing allocator/synchronization safeguards but remains hardware-unverified here. No permission to install a driver, download a model, or edit a checkpoint is implied.

### A real fixed-prefix example

On the pinned SmolLM2-135M CPU checkpoint, the prompt **“Gravity is”** produced three input tokens. At zero-based layer 15 and the final input token, these explicit edits matched separate direct-hook calculations for the retained vector bits and full-logit change metrics:

| Edit | Maximum absolute logit change |
|---|---:|
| Zero the attention-output vector | 1.646668 |
| Add 1 to MLP-output feature 0 | 0.038273 |
| Set block-output feature 0 to 0 | 0.193721 |
| Add 0 to attention-output feature 0 | 0 (bitwise no-op) |

The next-token choice stayed **` the` (ID 260)** in every case. Changed activations/logits do not necessarily change the selected token. All null/restoration controls passed; original capture/weight hashes, rotary bits and checkpoint files stayed unchanged. All 16 witness vectors survived export/reopen exactly. This is one short, specified comparison—not a semantic explanation or evidence about other inputs. [Exact current intervention record](../results/local-app-smollm2-135m-cpu-20261031/interventions.json).

With the already-downloaded pinned assets, reproduce the current workflow offline (the export directory must be new):

```bash
python results/local-app-smollm2-135m-cpu-20261031/verify_interventions.py --checkpoint ./models/smollm2-135m --export-root ./local-intervention-export
```

## Memory, failures and privacy

The default trace allowance is 128 MiB; the server has a 512 MiB ceiling, which can be lowered with `--max-trace-mib`. Byte/event estimates aggregate **all requested generation forwards**, including repeated prefix captures. A too-large requested capture is rejected before forwarding when it exceeds the calculated bound. Runtime limits still apply. Select fewer boundaries or shorten the request rather than expecting automatic thinning.

The model default preflight cap is 2 GiB of parameter storage. Neither that nor the trace allowance is a peak-RAM guarantee: loading copies, logits, temporary outputs, tokenizer/image processing, metadata and export/import copies cost extra memory. The separate GPU allowance defaults to 2 GiB. During CUDA load/run, the app temporarily applies the stricter of the existing CUDA caching-allocator fraction and this allowance divided by visible device memory, and restores the prior fraction afterward. This process-global allocator setting is intended for the dedicated, serialized app process, not a shared inference service. Free-memory/headroom checks happen before work; synchronized peak allocated/reserved values are recorded. T5/DistilBERT/ViT/Llama per-forward retained GPU captures are included in preflight planning, including ViT's wider intermediate vectors and uploaded/private pixel copies. `--max-vram-mib` limits the selectable allowance; `--max-model-mib` independently limits model parameter storage.

The allocator ceiling only covers PyTorch's CUDA caching allocator, **not all driver/library/context memory or other processes**. Free memory is an observation, not a reservation. Estimates cannot bound every kernel workspace, and GPU/CPU allocation can still fail. The app does not call `empty_cache` as proof that live work was safely freed. There is no asynchronous cancellation or general resource-isolation guarantee.

**Starting a new query replaces the previous RAM capture**, including when the new query later fails. The page warns before replacement; it does not show a confirmation dialog. Save first if you want to retain a run. Successfully loading another model also clears the current capture. **Unload model** releases the resident model without clearing the current capture, which remains inspectable/saveable. Clear releases the capture separately.

A failure publishes no partial trace as complete. If execution leaves model ownership uncertain, the app marks it quarantined and requires a process restart rather than pretending cleanup succeeded. Ordinary validation errors before forwarding can be corrected without a restart.

Save writes a **new**, named directory under the export root, containing `tensors.safetensors`, `manifest.json` and a completion status. Existing directories are never overwritten. Exports include text prompts/token IDs, answers, image-coordinate metadata and full captured activations; treat them as potentially sensitive. The default export directory is Git-ignored. The saved-run picker lists only immediate folders under the export root. It labels apparently complete entries **unverified** and incomplete entries unavailable; listing never reads the tensor payload or proves integrity. It is a bounded convenience for a trusted local filesystem, not protection against a hostile concurrent filesystem writer. You can also paste an export path. Open saved trace verifies file inventory, shapes, byte limits and the tensor digest, then loads original values into RAM. A missing or invalid export leaves the current capture, selected values, filters and comparison visible; only a successful Open replaces them. Checksums detect changed bytes relative to the manifest; they do not authenticate a model or establish execution provenance. Model paths are labels, not cryptographic weight attestations. The checkpoint and original uploaded image are not copied into the trace export; keep them separately if needed for replay.

The server uses a fresh session token, same-origin/Host checks and no remote assets or telemetry. These are browser boundary safeguards, not protection from hostile code already running as the same local user. Do not proxy it to the internet or share its session page.

## Verification

```bash
python -m pytest tests/test_app_runtime.py tests/test_app_cuda.py \
  tests/test_app_selftest.py tests/test_app_storage.py \
  tests/test_app_http.py tests/test_app_install.py tests/test_app_interventions.py \
  tests/test_app_intervention_http.py tests/test_app_demo.py tests/test_app_demo_http.py \
  tests/test_app_input_preflight.py tests/test_app_setup_http.py tests/test_transformer_trace.py \
  tests/test_distilbert_trace.py tests/test_flan_t5_trace.py tests/test_vit_trace.py \
  tests/test_probes.py tests/test_interventions.py tests/test_intervention_plan.py \
  tests/test_model_ownership.py tests/test_encoder_forward.py \
  tests/test_distilbert_intervention_workflow.py tests/test_vit_trace_comparison.py \
  tests/test_vit_intervention_workflow.py tests/test_llama_trace.py \
  tests/test_app_llama.py tests/test_app_llama_storage.py tests/test_app_llama_http.py \
  tests/test_app_llama_interventions.py -q
# Optional developer-only DOM/API contract check (Node.js; no browser install):
node tests/app_ui_contract.cjs
```

The application dependency set is verified here on Linux, Python 3.12.14 and CPU PyTorch 2.15.0; other operating systems/interpreter combinations were not runtime-tested. The core library’s older Python 3.10 declaration is separate from this app’s Python 3.11+ dependency requirement.

Tests use tiny locally generated checkpoints and synthetic tensors; they do not establish behavior for every pretrained model. The broader research suite and experimental opcode boundaries remain separate; see the [documentation guide](README.md).

The DOM contract check exercises mocked requests, not rendering. Real loopback HTTP and tiny local-model roundtrips were exercised separately. Chromium visual testing was blocked by this execution environment, so no browser-rendering or cross-platform visual validation is claimed.

### CUDA validation boundary

This executor has no visible NVIDIA device and a CPU-only PyTorch build. CUDA policy tests use explicit synthetic API responses; they are **not GPU execution results**. Hardware-gated tests in `tests/test_app_cuda.py`, `tests/test_app_llama.py` and `tests/test_app_llama_interventions.py` exercise tiny local GPT-2/T5/DistilBERT/ViT/Llama models, same-device boundary-bit/selected-token comparisons and CPU-offload/save checks when real CUDA is available. Run them in the CUDA environment, without hiding the device:

```bash
python -m pytest tests/test_app_cuda.py tests/test_app_llama.py tests/test_app_llama_interventions.py -q -ra
```

The focused app/adapter suite completed with **875 passed and 10 real-CUDA prerequisite skips** on the CPU executor. The UI DOM contract covered 226 mocked authenticated requests, including exact source filters, stable matched-vector navigation, stale responses, failed loads/opens, uncertain Open responses, demo onboarding and controlled interventions. Real loopback HTTP tests exercised all five families, source filtering through save/reopen, separate intervention witnesses and original-capture preservation. A freshly built wheel installed into a separate test directory also passed authenticated five-family source-filter/pagination checks: all 30 selected vectors retained their exact values, bits, indices and coordinates after unloading and reopening, with the new UI assets present. The Llama HTTP tests separately checked every vector across synthetic multi-head, grouped-query and static-Llama-3-RoPE models (126 vectors total), including BOS-token alignment and save/reopen. Static RoPE tests cross the original context length and return to a shorter prefix while checking unchanged frequency bits and exact full logits/captures. A tiny tied, sharded BF16-stored checkpoint also loads into float32 with exact conversion, tied-weight identity and unchanged source files. Those focused checks use synthetic checkpoints and do not establish pretrained quality or GPU behavior. The separately recorded real pretrained CPU check above covers one actual checkpoint. Demo tests cover no checkpoint writes/downloads, RNG preservation and synthetic provenance; installer tests use real interpreter/config inspection plus simulated install failures, without fresh dependency downloads. CPU encoder/ViT comparisons and DistilBERT/ViT intervention workflows remain included. The default historical-suite attempt again stopped during collection at the same 47 missing historical source/fixture errors in this sparse checkout; this is not an all-repository pass.

An additional real loopback check opened the previously verified SmolLM2 CPU export, attempted both a missing directory and a corrupt tensor digest, and verified that all 360 retained tensor hashes (1,620 vectors) and the capture revision remained unchanged. Save and a valid Open still worked afterward; this exercised retained pretrained data, not a new model execution.

A prerequisite skip is not a pass. No GPU speedup, VRAM measurement, rounding-gate fidelity, or pretrained-model quality has been established by this release's CPU-only validation.

### Run the standalone GPU check

In the installed app environment:

```bash
python -m looking_glass.app_selftest --device cuda:0 --max-vram-mib 1024
# The installed console entry point is equivalent:
model-looking-glass-gpu-check --device cuda:0 --max-vram-mib 1024
```

This is an explicit, bounded diagnostic, not a background test when the app starts. It creates tiny deterministic synthetic GPT-2/T5/DistilBERT/ViT/Llama checkpoints in a temporary directory, uses only local dependencies, and never reads a user checkpoint or downloads a model. The Llama fixture uses static `llama3` RoPE. All 25 ordered family/stage checks must pass for overall success. It checks actual device placement, same-device baseline versus traced full-logit bits, selected boundary bits, CPU offload, temporary export/reopen with token/patch coordinates, repeated runs, hook cleanup and allocator-policy restoration. A failure stops further device work when ownership is uncertain rather than reporting recovery as success.

Standard output contains a JSON report with overall and per-stage **passed**, **failed**, or **skipped** states. Exit codes are **0 = passed, 1 = failed, 2 = skipped/prerequisite unavailable**. An unavailable CUDA device/build is skipped, never a CPU substitute or GPU pass. Device memory allowance covers the scoped PyTorch allocator only, with the same external-memory limitations described above. Temporary diagnostic checkpoints/traces are not your model data; the command does not persist a report unless you explicitly redirect its output.

A passing check validates only these tiny synthetic workloads on that machine and runtime. It does not establish arbitrary-checkpoint compatibility, performance improvements, rounding-gate behavior, or hostile-code isolation. The larger hardware-gated pytest checks remain available separately.
