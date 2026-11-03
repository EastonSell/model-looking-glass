# Local model workbench

Run a model on your computer, then explore the values captured inside its layers. Use your own checkpoint on CPU or a compatible NVIDIA GPU. The tiny CPU demo is optional.

## Get started

1. Download **model-looking-glass-app.zip** from the [latest release](https://github.com/EastonSell/model-looking-glass/releases/latest), extract it, and open a terminal in the folder containing `install.py`. You need **Python 3.11+**.
2. Choose `--backend cuda` for a compatible NVIDIA GPU on Windows/Linux, or `--backend cpu` for CPU/macOS. On Windows run `py -3 install.py --backend cuda` (or `cpu`); on Linux/macOS use `python3`. The installer downloads packages, not drivers or model weights.
3. Open **Model & settings**, choose your local checkpoint and device, then **Inspect folder → Load model**. Enter text or choose an image, then **Run & capture**. No checkpoint yet? **Get a model** offers explicit downloads; **Load tiny CPU demo** needs none but is untrained.
4. Select a module in **Layers** and choose a token or image patch. **Exact values** opens the full vector. Each module initially selects its first captured vector, not an automatically matched token.

Next time, use `py -3 start.py` on Windows or `python3 start.py` on macOS/Linux. **Don't reinstall just to reopen the app.** If an older release has no `start.py`, use its direct launch command in the installation reference below. Stop the app with Ctrl+C; an active model run may need to finish first.

## Load your model

Open **Model & settings**, browse to a local model folder, choose **Inspect folder**, then **Load model**. Inspection checks the files; it doesn't prove the model will run. Use matching local safetensors weights and tokenizer/image-processor files. Compatible GPT-2, standard Llama, DistilBERT, T5 and ViT checkpoints are supported; see the exact compatibility table below. Or open **Get a model**, choose an explicit download, then use its folder. The catalog is small and curated—not a search of every Hugging Face model.

Start with a short input on your chosen device. CUDA requires a compatible NVIDIA GPU and driver. The project owner reports a successful GPU run; the automated validation environment remains CPU-only, and no GPU/driver details were supplied for that report. Memory allowances aren't a guarantee that a model fits. If a request is too large, shorten it or capture fewer layers.

## Keep a capture

**Save before another run or model load.** An accepted new run replaces the previous RAM capture even if execution later fails. Successful model replacement also clears it. **Unload model** keeps the capture available; **Clear** removes it.

Use **Save capture** to keep the full capture, then reopen it from the saved-run picker. Exports may contain private prompts, outputs and activations. They don't include checkpoint weights or the original uploaded image. Nothing is saved automatically. Keep the server local—don't share its session page or expose it online.

## Compare or try an edit

- **Across prompts:** keep a selected reference vector, run another prompt on the same loaded model, then compare a vector from the same module. The reference is RAM-only: Save capture does **not** save it or its comparison. Save the original capture before replacing it if you'll need it later.
- **Temporary edits:** with a new GPT-2 or supported Llama capture, select a vector and try zeroing it or changing a feature. This compares one fixed input, not a whole rewritten conversation. Checkpoint files and the original capture stay unchanged. Use **Save comparison** for the separate result.

Saved captures remain readable, but cannot authorize edits to a newly loaded model. If a run leaves the model unsafe to reuse, the app asks for a restart; ordinary input errors don't require one.

The displayed values are measurements, not named concepts or decoded thoughts. Output scores come from the complete model, not the selected hidden layer. The reference below explains these distinctions when you need more detail.

## Reference

Open only what you need. Original experiment records and detailed checks remain linked in these sections.

<details>
<summary>Install and start</summary>

## Install and start

### First installation: app ZIP or source checkout

Use Python **3.11 or newer**. For the smaller app download, choose **model-looking-glass-app.zip** in release **Assets**. GitHub's separate **Source code (zip)** includes research datasets and is much larger. Both contain Python source, not a standalone app or Windows executable. Choose **Extract All**, and open a terminal in the extracted folder containing `install.py`; do not run it inside the ZIP. No Git is needed for this route. Keep the folder in its final location: the installer creates a local `.venv` and an editable package installation that are not portable after moving the folder.

Windows PowerShell/Terminal:

```powershell
py -3 install.py --backend cpu
```

If `py` is unavailable but a supported Python is installed as `python`, use `python install.py --backend cpu`. If neither command exists, install supported Python first; this script does not install Python. On macOS/Linux use `python3 install.py --backend cpu`. For a compatible NVIDIA system, select `--backend cuda` for the initial installation instead. The optional tiny demo always uses CPU; load a real checkpoint to exercise your selected GPU.

Alternatively, with Git installed:

```bash
git clone --filter=blob:none --depth=1 https://github.com/EastonSell/model-looking-glass.git
cd model-looking-glass
python install.py --backend cpu
```

The installer creates this checkout's `.venv`, installs PyTorch 2.15.0 and application dependencies from official PyTorch/PyPI sources, and opens a loopback browser page. On Linux/Windows the default is **CUDA 13.2** (`torch==2.15.0+cu132`, from the [official wheel index](https://download.pytorch.org/whl/cu132/torch/)); this requires a compatible NVIDIA GPU and driver. It does not install or modify drivers. For a CPU-only system use `python install.py --backend cpu` (the default on macOS). The CPU build was exercised here; the CUDA build and hardware path were not. CUDA-capable PyTorch by itself does not establish a working GPU. Installing a Python project executes its packaging code; inspect a clone before trusting it. It does not install system services, configure credentials, or fetch models. Use `python install.py --dry-run` to inspect its commands, or `--no-launch` to install without opening a browser. An existing `.venv` requires explicit `--reuse-venv`; it is never silently replaced.

### Open again without reinstalling

On latest main, from that same folder:

```powershell
# Windows
py -3 start.py
```

```bash
# macOS/Linux
python3 start.py
```

`start.py` validates and uses only this folder's existing `.venv`. It does not create an environment, invoke pip, download anything, switch CPU/CUDA builds, or repair files. It rejects a missing, redirected, incompatible or moved environment and an installed app pointing to another checkout. The target interpreter runs in isolated mode so ambient `PYTHONPATH` does not redirect the application. A launch failure is reported as a launch failure, not a request to silently reinstall.

**Release compatibility:** `workbench-preview-2026-10-31-layer-map` and earlier releases do not contain this launcher; their tags remain unchanged. If `start.py` is absent, use the existing direct command:

```powershell
.\.venv\Scripts\python.exe -m looking_glass.app
```

On macOS/Linux the equivalent is `./.venv/bin/python -m looking_glass.app`. Do not rerun the installer just to open the app. A prepared custom environment can also use `model-looking-glass` after installing the package with `python -m pip install -e ".[app]"`; `start.py` intentionally targets the installer-created local `.venv`, not arbitrary activated environments.

Pass app options after `--`, for example `py -3 start.py -- --no-browser --port 8765` or `python3 start.py -- --models-dir /explicit/model/folder`. `python start.py --dry-run` prints the command without executing the environment or starting a server. These launch options do not change installed packages.

For an intentional package change, an existing CPU-only `.venv` can be explicitly changed with `python install.py --reuse-venv --backend cuda`; build-local version pins ensure pip does not silently retain the CPU build. Choose a compatible PyTorch build using its [official installation guidance](https://pytorch.org/get-started/locally/). This is not a driver installation or a necessary step for ordinary relaunch.

Options include `--no-browser`, `--port`, `--models-dir /explicit/model/folder`, `--export-dir /chosen/export/folder`, and `--download-dir /chosen/model/folder`. `--models-dir` sets an explicit browsing root. Without it, choose **Browse folders** with an empty path for normal local starting locations, or enter your own root. Offering those locations does not enumerate their contents: a folder is read only when selected. Browsing automatically classifies the selected folder and its immediate visible subfolders from bounded config metadata. The browser stays inside its chosen root, does not recursively scan home/drives, and never loads weights or tokenizers while browsing. Large folders are bounded and show incomplete/truncated results. See [finding local checkpoints](#finding-local-checkpoints). The server binds only to `127.0.0.1`, chooses a free port by default, and prints its URL. Keep the terminal running. Stop with Ctrl+C; a long active forward may need to finish first. This is a trusted single-user local tool, not a public service.

### If installation stops

Before installing packages, the installer checks the **environment's own Python**, not just the Python running `install.py`: it must be 3.11+, belong to this checkout's isolated `.venv`, and have pip. System-site-package environments and redirected `.venv` roots are rejected; a normal `bin/python` symlink to the base interpreter is allowed. `--dry-run` remains read-only and does not execute the environment.

Dependency commands ignore ambient pip/Python redirection variables and pip configuration files, require a virtual environment, and use the explicit official indexes shown by `--dry-run`. This prevents an unrelated `PIP_TARGET`, prefix or extra-index setting from silently redirecting this installation. Your environment and configuration files are not modified. This is install-target hygiene, not a sandbox for trusted package/build code.

An unsuccessful dependency step identifies which stage stopped and prints a copyable, explicit resume command using the same CPU/CUDA backend. Installed packages may remain, but the app is not launched and success is not reported. Resume uses `--reuse-venv --no-launch`; start the app separately after installation succeeds. No automatic retry, backend switch, deletion or environment recreation occurs.

If environment creation or its interpreter check fails, the existing directory is preserved. An older Python, missing pip or a moved/broken environment needs repair or a fresh environment; blindly adding `--reuse-venv` does not fix it. Keep any files you need, then use a fresh trusted checkout with supported Python rather than deleting an unknown environment. If installation completed but browser/app startup failed, the installer distinguishes that from a package failure and prints the direct app-start command.

</details>

<details>
<summary>Download a public model</summary>

## Download a public model

Open **Model & settings → Get a model**. Choose CPU or an available CUDA device and an estimated context length. Review the download size, model card/license and destination, then choose **Download**. Merely opening this list does not contact Hugging Face. The initial catalog contains two pinned, public Apache-2.0 snapshots:

- **SmolLM2 135M:** about 260 MiB to download. This exact checkpoint has previous CPU results in this repository; that is not a GPU or quality guarantee.
- **DistilGPT2:** about 339 MiB to download. Its public metadata and file hashes were checked, but this download path has not been validated by loading it. Use **Inspect** and **Load** to check your installation.

The list sorts by estimated loaded memory and normally hides models over a known device, server or context limit. **Show models over the estimate** reveals them. Unknown memory stays labelled unknown; CPU RAM is not measured. These estimates use float32 weights, eager forward work, capture storage and headroom—not just the download size or total VRAM. They use server limits, **not your selected Load GPU allowance**, so a smaller allowance can still reject a model. Free memory can change, longer runs need more capture RAM, and estimates do not promise that a model will fit or run.

Progress, the current model ID and its completed folder are shown. **Use downloaded folder** fills and focuses the checkpoint path in **Model & settings**, ready for you to review: it does not inspect or load the model or replace your capture. Background progress and completion never move focus. Choose **Inspect folder**, then **Load** yourself. A completed download verifies the pinned files, not successful inference. Downloads default to `looking-glass-models/`; set `--download-dir` when starting the app to choose another location. Downloaded weights are not part of capture exports.

Only an explicit Download request uses the network. It fetches the reviewed files from an immutable revision, checks every size and hash, and publishes a new folder only when complete. It never overwrites an earlier download. No access tokens, gated models, custom Python code, pickle weight files or arbitrary repository URLs are accepted. Public source availability and network restrictions can still make a request fail.

**Cancel download** stops between bounded reads; it can also wait behind an active model run. If status becomes uncertain, use **Refresh download** before acting again—there is no automatic retry. Ordinary failure/cancellation removes only that job's partial folder. Closing the app abruptly can leave a private partial directory; there is no automatic resume or deletion of other files. A finished download stays on disk until you remove it yourself. After restarting, open **Get a model → Browse downloads** to reuse it. Open the download folder, then its `model` subfolder, and choose **Use path** before Inspect/Load. This uses the existing bounded folder browser; a configured `--models-dir` still limits navigation. If it excludes the download location, use a known checkpoint path directly or restart with the intended browsing root—browsing never expands that root silently. Model execution remains local and offline; prompts and captures are not sent to the registry.

</details>

<details>
<summary>Finding local checkpoints</summary>

## Finding local checkpoints

Use **Model & settings → Model options & compatibility → Browse folders**. With no path and no configured model root, the app offers ordinary local locations (including an existing Hugging Face cache, and drives where the platform supports listing them). Choose one to begin; no background whole-PC scan runs. You can also paste a known folder. A configured `--models-dir` remains the browser boundary.

Each visit checks the **current folder and immediate subfolders only**. It reads bounded `config.json` metadata and checks for expected asset filenames; it does not parse weights, execute checkpoint code, import Transformers, install packages or contact a registry. The status distinguishes a supported-family/head **candidate**, unverified head metadata, unsupported configurations, incomplete exports and unreadable files. A candidate is not verified loadable: dimensions, weight headers, input assets, runtime/device constraints and actual inference still require **Inspect** and then explicit **Load/Run**. Checkpoint and tokenizer provenance are not attested by a filename.

Discovery stops at 500 directory entries and 200 visible folder results, with 64 KiB per config and 2 MiB aggregate config reads. Hidden folders are not enumerated as children; a known hidden cache can be selected explicitly. It does not follow directory symlinks/junctions to scan elsewhere. Config links outside the chosen root are not read; choose an appropriate cache parent as root or use explicit Inspect on a known snapshot. Permission, malformed-file and truncation notices remain local and do not discard the loaded model or retained capture. These are count/byte bounds, not a guarantee about slow or disconnected filesystem latency.

</details>

<details>
<summary>First capture without a checkpoint</summary>

## First capture without a checkpoint

Choose **Load tiny demo** to construct a fixed, tiny GPT-2 and byte-level tokenizer entirely in CPU RAM. No weights, tokenizer files or temporary checkpoint are written or downloaded. It is **random and untrained**: its output is not a useful answer or evidence of learned language, semantic features, model quality or GPU performance. The demo always uses CPU, independently of your local-checkpoint device selection.

Loading it replaces the loaded model and retained capture/comparison after success, so explicitly save anything you want first. The demo starts with auto hooks and short token limits; memory/event allowances and your checkpoint path/device preferences are unchanged. A suggested prompt fills only an empty prompt field; choose **Run** yourself. Then select an event and source position to inspect full values/bits, or try a controlled activation edit. The ordinary resource limits, exact capture and export paths still apply. The tiny model has a 128-position context window; byte-level tokens make ordinary text longer in tokens than a trained tokenizer would.

The demo's provenance, seed/configuration and untrained warning stay attached to captures and comparisons, including saved/reopened exports. A saved capture is not a saved model. Loading a fresh demo creates a new live-model binding; old or reopened captures cannot authorize edits to it. To use your own model afterward, load a supported local folder normally. Nothing about this demo broadens checkpoint compatibility.

</details>

<details>
<summary>Checkpoint compatibility</summary>

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

Llama support is deliberately bounded to the installed Transformers 5.22.1 standard eager `LlamaForCausalLM` with `model_type="llama"` and `pretraining_tp=1`. Ordinary RoPE (`rope_scaling` absent or null) and the exact static `rope_type="llama3"` frequency transformation are supported. Standard multi-head and grouped-query attention are checked against actual projection dimensions. Linear, dynamic, YaRN, LongRoPE, partial rotary dimensions and custom implementations remain rejected rather than silently changed. Use matching local fast-tokenizer assets (`LlamaTokenizerFast`, standard GPT-2 BPE or compatible fast JSON assets); local model loading does not download SentencePiece conversion dependencies, weights or chat templates. The prompt is supplied as raw text: an instruction-tuned checkpoint may need its own correctly formatted prompt. The app does **not** apply its chat template automatically, and only fresh privately bound captures enable the fixed-prefix intervention panel described below.

For static Llama 3 RoPE, the legacy `rope_scaling` configuration must contain exactly `rope_type`, `factor`, `low_freq_factor`, `high_freq_factor` and `original_max_position_embeddings`. Native Transformers 5 `rope_parameters` adds `rope_theta`; conflicting legacy/native forms are rejected. Factors must be finite (`factor >= 1`, `0 < low_freq_factor < high_freq_factor`); the original context length must be a positive integer below `max_position_embeddings`. The adapter checks the standard initializer and unchanged rotary-buffer bits across each forward. It uses the [pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/modeling_rope_utils.py), not a replacement approximation. A large configured context is not a promise it will fit: uncached prefix work, full logits and every selected vector still count against the app limits.

Float32 needs four bytes per parameter before captures, logits and loading copies: a billion parameters alone is about 3.7 GiB, beyond the default 2 GiB model ceiling. For a larger compatible checkpoint, set `--max-model-mib` explicitly and, on CUDA, `--max-vram-mib` plus the Settings allowance only when your machine has sufficient headroom. Increasing these limits does not create memory or guarantee a successful load.

</details>

<details>
<summary>Verified pretrained example: SmolLM2-135M (CPU)</summary>

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

</details>

<details>
<summary>Device selection and GPU semantics</summary>

## Device selection and GPU semantics

**Auto** prefers available CUDA for all five supported families and chooses CPU when CUDA is unavailable, reporting why. An explicit `cuda:N` request never silently falls back; unavailable devices and unsupported model families are rejected. Choose the device beside the checkpoint path. It applies to the next **Load model**, not the model already loaded; the status and Run tooltip show that model's actual device. Device indices follow PyTorch visibility (including `CUDA_VISIBLE_DEVICES`), not necessarily physical card numbering.

| What you see | What to do |
|---|---|
| CPU after choosing CUDA | Save your capture, then **Load model** again; the selector does not move an already-loaded model. The tiny demo always uses CPU. |
| CUDA unavailable or Auto chooses CPU | Run the [GPU check](#run-the-standalone-gpu-check) with this app's `.venv`, not another Python. A CPU build cannot run CUDA; only intentionally [change the installed backend](#open-again-without-reinstalling). A CUDA build still needs a compatible visible NVIDIA GPU and driver. |
| GPU check exits with code 2 | Read the JSON reason: prerequisites are unavailable, not a passing GPU result. CPU-only users can skip this GPU check and load with **CPU**. |
| Memory allowance or out-of-memory error | Try a smaller model, shorter input or fewer captured layers. Model, trace and GPU allowances are separate; raising a limit does not create memory. Restart if the app says the model is unsafe to reuse. |

The app uses **direct forward hooks**: it copies the boundary values that the model actually computed. It does not implement a rounding-based gate, reconstruct hidden values from rounding error, or use the experimental auxiliary-gradient reporting channel. GPU arithmetic can differ from CPU arithmetic even at float32 (kernels, reduction order, TF32 settings and subnormal behavior); exact offload/save means preserving the observed values, **not** promising CPU/GPU-identical outputs. The run records device and numeric settings rather than silently changing global precision settings. See the [gradient-channel counterexample](GRADIENT_DENORMAL_V1.md).

GPU captures are moved synchronously to CPU RAM without dtype reduction. GPT-2 offloads at each selected hook; T5, DistilBERT, ViT and Llama first retain one forward's complete selected captures on the GPU, then the app offloads and releases them. DistilBERT predicts all supplied mask positions in one forward, not autoregressive generation. This is deliberately conservative and can be slower than asynchronous/pinned-memory pipelines. No lossy compression, sampling or dropped vectors is hidden in the transfer.

The standalone DistilBERT adapter remains CPU-only by default. Its explicit `synchronous_cpu=False` opt-in permits one CUDA float32 device, validates inputs, logits and captures on that device, and synchronizes before returning. CUDA results deliberately have no CPU `EncoderForwardContext`: the historical CPU comparison contract is not asserted for a GPU run. Callers must still own the model and handle failed/uncertain CUDA execution; the app quarantines such failures.

ViT image decoding and preprocessing remain on CPU, with bounded RGB output geometry checked before transfer. Its CUDA adapter hashes the exact processed pixel encodings through blocking integer-bit CPU copies before/after forwarding; it does not infer original-image coordinates or change pixel precision. Like DistilBERT, standalone `trace_vit_classifier` is CPU-only by default and accepts CUDA only with `synchronous_cpu=False`. Its historical CPU-only comparison API remains unchanged and rejects CUDA-origin events, even after their vectors are offloaded.

</details>

<details>
<summary>Full capture and readable inspection</summary>

## Full capture and readable inspection

**Fit view** fits the captured layers back into the 3D view without changing the selected vector or reading more data. Use **List** when WebGL is unavailable.

1. Browse an explicitly chosen local folder or paste a checkpoint path. Inspect the model, select Auto/CPU/CUDA and the GPU allowance, then load it. Choose **Auto: all supported points**, or explicitly select roles and layers.
2. Set input/output-token limits and capture byte/event allowances. Prompt truncation is rejected, never silently performed.
3. Enter a prompt or image and run. A successful trace is exposed only after every selected capture and adapter cleanup completes. The run records its own model label/settings, input IDs and forward-step coordinates, independently of later model selections.
4. Search across the entire capture by module path/kind or layer, or filter by exact **source position**, **forward** and **stack**. The Source column labels verified single-vector events; multi-vector events show a count rather than pretending to represent one token. Selecting a filtered match opens its actual vector, even when it is not vector 0. Positions are zero-based: T5 decoder position 0 is the start token; ViT position 0 is CLS, followed by row-major processed-image patches. Module search does not search tokenizer text. Clear filters to reveal unknown-source events, which cannot match exact source filters. Use the paginated **source-position picker**, previous/next vector buttons, or exact flat vector index. The viewer shows the recorded forward, layer, batch and token position/ID; T5 encoder and decoder streams are distinct. ViT shows CLS or patch coordinates and the processed-image pixel box, not original-image pixels. The raw table includes float32 values and hexadecimal bits, preserving distinctions such as signed zero. Pagination is presentation only: all captured values remain in RAM.
5. Optionally switch to **grouped coordinates**. This displays contiguous coordinate means and ranges; it is an explicitly approximate display, **not semantic concepts or inferred reasoning**. It does not replace, compress, or discard the original tensors.

Navigation uses only the capture’s recorded shape, selected source positions and matching forward inputs. Inconsistent or missing provenance is labeled **unknown**, while raw vectors remain available. Older exports may show exact token IDs with labels unavailable. Tokenizer pieces and single-token decoded fragments are captured from the run’s tokenizer, not a later loaded model; they are not necessarily words or contextual text. These optional display labels are capped at 512 characters each and 256 KiB total in compact UTF-8 JSON, with truncation/omission reported. This never truncates token IDs or activation tensors. A selected output token from forward N first becomes an input position at N+1, if that later forward actually ran; the final generated token may therefore have no captured input activation. Internally consistent coordinates are not independent proof of an imported export’s provenance.

Browsing reuses shared forward-input validation **only within one read request**; every event still undergoes its own source/shape/identity checks, and nothing is cached across reads or saved in exports. On this Linux/Python 3.12 CPU executor, a paired synthetic-catalog benchmark (six layers, 512 tokens, width 768; seven alternating measurements after warmup) reduced a 9,216-event DistilBERT stack filter from **414 ms to 73 ms median**. A GPT-2 100-vector picker page fell from 9.1 ms to 5.3 ms. Complete responses and tensor-byte hashes matched; ViT saw little change. These are storage-method timings, not browser rendering, inference or GPU speedups, and are not performance guarantees.

Auto means every supported **module boundary**, not every mathematical operation, neuron state, attention probability, or internal node. T5 cross-attention is not a separate capture role in the current adapter. GPT-2/T5/Llama use explicit uncached generation forwards: later steps recompute the growing prefix, and repeated token vectors belong to distinct forwards. Attention/MLP/block changes are observations, not a proof of a unique causal mechanism or complete interpretability.

</details>

<details>
<summary>Reading tokens and model output</summary>

## Reading tokens and model output

The visual explorer separates **what went in**, **what the complete model predicted**, and **the hidden vector captured at a named module**. These are related execution records, not interchangeable data types.

Choose a recorded forward to read its input-token strip and output readout. Text chips show retained individually decoded token fragments when available, with exact vocabulary IDs and zero-based positions; raw tokenizer pieces remain available in details. Byte-level tokens can split a character or word, show replacement characters, or include whitespace. Joining individually decoded fragments is not guaranteed to reconstruct the prompt; the original prompt and model answer remain separate.

Selecting an input token applies the existing exact source-position/forward/stack filters to the capture catalog. It does not run the model, invent an activation, or discard other vectors. T5 encoder input and decoder input are distinct streams; decoder position zero is its start token. ViT uses image/patch coordinates, not text tokens. Missing or inconsistent recorded labels stay unavailable rather than using a different loaded tokenizer.

For GPT-2, Llama and T5, the output readout shows the recorded greedy selection and top next-token **logits** for that full forward's last readout position. Logits are scores, not percentages; the displayed top candidates are not renormalized into a misleading partial-vocabulary probability distribution. DistilBERT shows its simultaneous mask predictions, while ViT shows recorded image-class probabilities. These are complete-model outputs, **not predictions made by the selected intermediate layer or individual feature**.

The source inspector also offers **Recorded token evidence** when the saved output belongs to that exact position. For GPT-2/Llama this is the final unmasked input position; for T5 it is the final decoder position. DistilBERT can show a prediction at a recorded mask position. Earlier text positions, T5 encoder positions and ViT patches do not acquire guessed token predictions. A missing or contradictory saved record is marked unavailable. This uses the capture's own forward IDs, source IDs and retained scores—not the currently loaded tokenizer or another run. It performs no new forward pass and computes no probabilities.

The next token selected by forward N becomes an input activation position only if forward N+1 ran. The final selected token therefore often has no captured input vector. Output tokens are displayed separately from selectable input positions. An unchanged next-token choice does not mean all scores or activations were unchanged.

For example, the previously verified pretrained SmolLM2 run of “Gravity is” selected “ the”, then “ force”, “ that”, and “ holds”. The first forward's “ the” score was 22.05638885498047; it is a logit, not a 22% probability. The new display can show that existing recorded evidence without rerunning the checkpoint. The output “ holds” is not an input position in that run's last captured forward. See the unchanged [CPU record](../results/local-app-smollm2-135m-cpu-20261031/record.json).

**A hidden feature value is not a vocabulary ID.** Tokenizers cannot turn arbitrary float vectors into words. This view makes actual source tokens and observed output choices readable; it does not assign semantic names to neurons, decode an intermediate thought, or implement a logit lens. Exact raw values/bits, grouped display, references and controlled edits remain accessible.

</details>

<details>
<summary>Optional layer map</summary>

## Optional layer map

The **Layers** view shows recorded module outputs. The central scene uses genuine Three.js planes, with a compact selected-vector inspector on the left and a narrow prompt/response panel on the right. Run stays beside its prompt or image input. The floating scene toolbar switches 3D/List and fits the current layers; it does not run the model. Layer labels remain visible; module labels appear only for the selected or hovered site to keep dense captures readable. Hover a band with a mouse or pen to preview its existing label before clicking. Hover uses the current page’s captured metadata, does not move the camera or select/read a vector, and clears while dragging or leaving the scene. Pointer updates are coalesced into the next render frame; no idle loop is added. The warm highlight identifies the selected captured site, not activation sign or importance. Drag to orbit, scroll to zoom, or use the native **List** buttons instead. Choose a forward and page through captured layers. There are no invented neuron dots, connections, activation-derived positions or semantic clusters. Only captured sites appear—not uncaptured layers inferred from configuration.

The thin-edged surfaces take visual inspiration from [Neuron Explorer](https://github.com/lyramakesmusic/neuron-explorer); this renderer is independently implemented. Pinned Three.js, OrbitControls and fonts are bundled locally with their licenses and [asset provenance](../looking_glass/app_static/vendor/provenance.json). Viewing the scene makes no CDN request. The font/library archives were checked against official registry integrity values before their assets were copied.

Each page has at most 12 layers and 72 captured sites. No idle animation keeps the browser busy. Reduced-motion preferences disable camera transitions. If WebGL is unavailable, the app explicitly switches to the captured-site list; that is a fallback, not simulated 3D. Native buttons remain a keyboard alternative even when 3D works. Collapse either dock to give the layers more room; the view refits between the remaining panels or toggle rails. Bracket shortcuts work only outside text fields. Selecting an observation in the exact table reopens its inspector. On narrow screens, panels remain below the scene and do not reserve horizontal space. Changing the view does not run inference, change vectors, retokenize inputs or save data. Missing metadata stays unavailable rather than being replaced with decoration.

Selecting a module opens its **first captured event at vector zero** in the existing exact-value viewer. Use the source-position picker to choose another token or image patch. Representatives are not automatically token-aligned across sites, and this map is independent of the table's search/source filters. Supported fresh GPT-2/Llama selections can use the same reference comparison and controlled-edit tools; the map does not broaden their eligibility. Saved captures can be mapped without a loaded model, but do not acquire live edit authority.

This is a **recorded-boundary diagram**, not a neuron graph, attention visualization, semantic projection or causal explanation. Spacing, grouping and colors are presentation only; no connections or activation magnitudes are inferred. It reads metadata, not tensor statistics, and preserves every original value. Invalid or unsupported provenance is counted as unmapped and remains available in the exact event table. Explicit forward/count labels distinguish other-forward and missing-forward events. Native disclosure controls and module buttons provide standard keyboard interaction; the original checks were structural and simulated, not browser or screen-reader validation. See the verification records below for later checks.

The installed-wheel CPU verification on the pinned SmolLM2 checkpoint mapped **30 layers / 90 observed module sites per forward** across two captured forwards. All **180 representative vectors × 576 features** had exact retained bits, before and after Save/Unload/Clear/Open; hashes of every retained tensor, model parameters/buffers and checkpoint files were unchanged. A mapped live Llama selection also used the existing reference comparison without changing eligibility. See the [record and reproducible offline check](../results/local-app-smollm2-135m-cpu-20261031/layer_map.json). This verifies metadata-to-payload navigation—not visual rendering or semantic interpretation.

```bash
python results/local-app-smollm2-135m-cpu-20261031/verify_layer_map.py --checkpoint ./models/smollm2-135m --output ./layer-map-check.json
```

</details>

<details>
<summary>Try a controlled activation intervention</summary>

## Try a controlled activation intervention

With a **fresh live GPT-2 or supported standard Llama capture** and a known selected vector, use the intervention panel to **zero that vector**, **add to one feature**, or **set one feature**. These are temporary runtime operations, never checkpoint/weight edits. Imported captures, another model load, unknown coordinates, and other architectures are read-only for this workflow. Reloading the same path does not restore the private live-model binding; make a fresh capture. Stale selections from another browser tab are rejected.

The experiment replays exactly one recorded input prefix and its attention mask, not the whole generated continuation. It runs an unhooked baseline, a same-site null control, the treatment, and an unhooked restoration. The null/restoration full logits must match the baseline bitwise; the fresh selected source must match the original capture. Other positions at the selected boundary must remain unchanged, and single-feature edits must preserve the other feature bits. Llama additionally checks static rotary-buffer/configuration identity and bits, unchanged exact input IDs/mask, full same-device logits and absence of a KV cache on every trial. A failed control publishes no successful comparison. This establishes a bounded same-input intervention comparison, **not** a model explanation, backdoor detector, general repair, or comparison between diverged generations.

The original RAM capture stays intact. The separate panel shows baseline/treatment next-token choices, a compact same-token probability comparison, explicit no-op results, and four full selected-vector witnesses (null before/after and treatment before/after), with exact value/bit pagination. It is **not a full-model post-intervention trace**; full logits are compared but only metrics/top candidates are retained in the report. Choose **Save comparison** to export this separate result; the original Save still saves the original capture. Reopening a comparison permits inspection, not another live intervention; expand its run details to read the saved controls and metrics.

The compact table aligns the union of each trial’s top five token IDs (at most ten rows), showing baseline/edited percentages and the signed change in **percentage points**. Probabilities use a stable float64 softmax over the **entire next-token vocabulary**, not just the displayed tokens; visible rows need not sum to 100%. These are model diagnostics, not calibrated confidence or correctness. Display percentages are rounded; exact logits, log-probabilities, deterministic full-vocabulary ranks (token ID breaks ties), full-logit metrics and whole-vocabulary total variation remain in expandable metadata/export. Float64 probabilities can underflow to zero; finite log-probabilities remain available. Old saved comparisons without compatible metadata show their original logits and explicitly mark probabilities unavailable—nothing is inferred from a top-token list. Opened comparisons get the same compact read-only view without acquiring a live experiment binding.

### Run an automated component-sensitivity experiment

After a fresh live GPT-2 or supported Llama capture, select an input vector, open **Experiment**, and choose **Run an experiment · component sensitivity**. The sweep uses that vector's recorded forward and input position for every test. Choose zero-based, comma-separated layers and any combination of attention, MLP and block output sites; leaving layers blank selects up to six captured layers. Capture the relevant sites first (the default Auto capture is a convenient starting point). Missing or unsupported sites are reported rather than treated as zero effects.

Each selected site temporarily zeros the complete vector and uses the same four-forward, fixed-prefix intervention and controls described above. Before ranking sites, the sweep compares an internal SHA-256 digest of the full baseline logits across trials; a mismatch stops the sweep and makes the report inconclusive. The digest is not saved in reports. The existing intervention limit of 64 input tokens also applies here. A maximum of 18 site tests means at most 72 model forwards; the page estimates this count before starting. The loaded model runs on its selected CPU or GPU. Runtime and memory use depend on the model, input and device, so start with fewer sites for a larger model. The original capture and any manual comparison remain separate and unchanged.

Experiments currently have no cancellation control. A submitted request may still run after the tab closes or loses its connection, even if it was waiting for another operation. Other app actions wait until the current request finishes. The same applies to repeated tests and cross-prompt patches.

The layer-by-layer report lists each site's status, the change in the baseline top token's full-vocabulary probability in percentage points, total variation across the vocabulary, and the maximum last-position logit change. Expand the details for controls and exact evidence, or use **Download report JSON** to save the report locally. A failed control or interrupted test is not a measured zero; read the reported status before interpreting a row. A large change only describes the result of this specific zeroing operation for this input and position. It does not identify a concept or establish that the finding generalizes to other prompts.

#### Compare prompts with repeated tests

In the same experiment controls, open **Compare prompts with repeated tests**. Enter two to four prompts, one per line, choose the layers and components above, and select **Run repeated tests**. Each prompt gets a fresh temporary capture on the same live model, and every test zeros the final input-token vector at one selected site. The original capture, retained reference, and manual comparison remain available. The report names each prompt's actual final input token and position and shows its own baseline token, probability change, full-vocabulary total variation, and control status.

The repeated test allows at most 12 prompt/layer/component tests and 52 model forwards, including the temporary captures. Blank layers select as many captured layers as fit that limit, up to six. Each prompt is limited to 512 characters and 64 input tokens; long inputs are rejected rather than truncated. Runtime and memory use vary by model and device. A failed or inconclusive prompt is identified, and later prompts are skipped. The report is temporary until **Download repeated-test JSON**; it includes prompt text and token IDs, so treat the downloaded file as potentially sensitive.

This is a small set of within-prompt interventions, useful for exploring whether a change appears on the supplied examples. Final tokens may refer to different words across prompts, and each prompt can have a different baseline top token; the app does not align meanings or claim statistical significance or general robustness. No activation is transferred between prompts in this repeated test. The separate reference-vector view compares recorded values across prompts without patching.

#### Patch an activation between prompts

Open **Patch an activation between prompts** in the experiment controls. Enter a donor (clean) prompt and a receiver (corrupted) prompt, then choose **Preview tokens**. Explicitly select one zero-based token position from each preview, choose a layer and component, and select **Run activation patch**. The labels “clean” and “corrupted” describe the inputs you chose; the app does not determine whether either is correct. Editing a prompt requires another preview.

This uses two temporary captures from the same live model and inserts the donor vector into the selected receiver position at the same layer and component. The clean baseline, unchanged receiver baseline, and patched receiver are reported for the clean baseline's top token, with full-vocabulary normalized probabilities; the receiver distribution table also shows its top-token union. A donor same-vector replay, receiver null and restoration, fresh source vectors, the patched vector, and other positions are checked bitwise. The existing capture, reference, and manual comparison stay separate. The test uses ten model forwards; prompts are limited to 512 characters and 64 tokens each, and ordinary memory limits still apply.

The chosen tokens are not automatically aligned by meaning. A change from one patch is evidence about that intervention on those inputs, not a complete model explanation or proof of generalization. The separate report contains prompt text and token IDs and remains in RAM until **Download patch report JSON**; store the download privately.

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

The normalized-probability follow-up shows why raw logits and probabilities are different: in the zero-attention case above, token 260’s logit fell from **22.0564 to 21.4455**, while its full-vocabulary probability rose from **18.5593% to 19.2629%** (**+0.7036 percentage points**). It remained the selected token. Independent scalar `exp`/`math.fsum` normalization agreed within the recorded tolerances; the no-op probability changes were exactly zero. Saved probability metadata reopened unchanged. [Probability verification record](../results/local-app-smollm2-135m-cpu-20261031/probabilities.json).

```bash
python results/local-app-smollm2-135m-cpu-20261031/verify_probabilities.py --checkpoint ./models/smollm2-135m --export-root ./local-probability-export
```

</details>

<details>
<summary>Compare a vector across prompts</summary>

## Compare a vector across prompts

After a live GPT-2 or supported Llama capture, select an observation and vector. Open **Reference vector → Keep selected vector**, then run another prompt on the **same loaded model instance**. Select a vector at the same module/layer and choose **Compare selected vector**. Both input records and token/forward positions stay explicit: there is no automatic word alignment, concept labeling or causal explanation. Different positions are allowed, but module, output selector, width, dtype, execution device and recorded numerical settings must match.

The table pages every feature's unchanged float32 values/raw bits and a float64 `current − reference` difference. Display and reference arithmetic decode IEEE float32 integer fields into float64, preserving signed zeros and subnormals even when the caller enables CPU flush-to-zero; they do not change that mode. Full-vector distance and cosine are numerical diagnostics, not semantic similarity or feature importance. A zero norm makes cosine unavailable. Bitwise and numerical changed counts are separate: signed zeros can have different bits while their numerical difference is zero.

The ordinary approximate grouped view also decodes source bits before computing means/ranges. It is still lossy: bins do not preserve individual features. Rows larger than 1 MiB use bounded chunk sums, with explicitly disclosed float64 summation-rounding differences; exact values and raw bits remain separately accessible.

Only one vector is pinned, at most 1 MiB, with at most 256 KiB source metadata. Its tensor bytes count against the shared server trace allowance, including new captures and intervention witnesses. Oversized requests reject rather than truncate; bounded float64 arithmetic workspace is separate, and these limits are not peak-process-memory guarantees. Failed replacement/comparison preserves the pin. Clear capture does **not** clear the reference; **Clear reference** does. After unload, reload, quarantine or opening an imported trace, the pin remains readable but comparison requires its original live-model binding—it is not authorized by a matching checkpoint path.

**The reference is RAM-only and has no separate export. Save capture saves only the current full capture, not the pin or vector comparison.** Save the original full capture before another prompt if you need it later; closing the app loses the pin. Keeping another vector explicitly replaces it. This is a bounded inspection aid, not a two-run history system.

The [pinned SmolLM2 CPU record](../results/local-app-smollm2-135m-cpu-20261031/reference_vectors.json) compares the final input token `is` at `model.layers.15.self_attn` for `Gravity is` versus `Electricity is`. All 576 feature values/bits and float64 deltas matched independent scalar calculations; L2 distance was 11.56439 and cosine 0.876129. These are numerical observations, **not evidence of semantic similarity**. Both 270-vector captures, model parameters and checkpoint assets remained unchanged. Reproduce without downloading anything:

```bash
python results/local-app-smollm2-135m-cpu-20261031/verify_reference_vectors.py \
  --checkpoint /your/pinned/local/SmolLM2-135M --output /new/reference-check.json
```

The record also includes a bounded paired CPU display timing check: seven alternating samples after warmup, 256 displayed values/bins. Exact raw decoding added about 0.03 ms per page in these synthetic cases. A 300,000-feature grouped view took 5.79 ms versus 2.37 ms before the fidelity fix; decoded chunks are reused and bounded. This is a measured tradeoff, not a browser/GPU/inference speedup. With FTZ off, smaller-row responses and all fixture tensor hashes were unchanged; the large-row response adds its rounding note.

</details>

<details>
<summary>Memory, failures and privacy</summary>

## Memory, failures and privacy

The default trace allowance is 128 MiB; the server has a 512 MiB ceiling, which can be lowered with `--max-trace-mib`. Byte/event estimates aggregate **all requested generation forwards**, including repeated prefix captures. A too-large requested capture is rejected before forwarding when it exceeds the calculated bound. Runtime limits still apply. Select fewer boundaries or shorten the request rather than expecting automatic thinning.

The model default preflight cap is 2 GiB of parameter storage. Neither that nor the trace allowance is a peak-RAM guarantee: loading copies, logits, temporary outputs, tokenizer/image processing, metadata and export/import copies cost extra memory. The separate GPU allowance defaults to 2 GiB. During CUDA load/run, the app temporarily applies the stricter of the existing CUDA caching-allocator fraction and this allowance divided by visible device memory, and restores the prior fraction afterward. This process-global allocator setting is intended for the dedicated, serialized app process, not a shared inference service. Free-memory/headroom checks happen before work; synchronized peak allocated/reserved values are recorded. T5/DistilBERT/ViT/Llama per-forward retained GPU captures are included in preflight planning, including ViT's wider intermediate vectors and uploaded/private pixel copies. `--max-vram-mib` limits the selectable allowance; `--max-model-mib` independently limits model parameter storage.

The allocator ceiling only covers PyTorch's CUDA caching allocator, **not all driver/library/context memory or other processes**. Free memory is an observation, not a reservation. Estimates cannot bound every kernel workspace, and GPU/CPU allocation can still fail. The app does not call `empty_cache` as proof that live work was safely freed. There is no asynchronous cancellation or general resource-isolation guarantee.

**An accepted new query replaces the previous RAM capture**, including when execution later fails. Save first if you want to retain a run; the replacement warning is not a confirmation dialog. Run keeps the existing view while the request is pending. If an early rejection leaves the server’s capture revision unchanged, its selected values, filters and comparison stay visible. If the server cleared or replaced it—or a network failure makes the outcome unverifiable—the UI hides stale capture values instead of labeling them as current. A successful new run remains the current result even if a subsequent event-list request fails. Successfully loading another model also clears the current capture. **Unload model** releases the resident model without clearing the current capture, which remains inspectable/saveable. Clear releases the capture separately.

A failure publishes no partial trace as complete. If execution leaves model ownership uncertain, the app marks it quarantined and requires a process restart rather than pretending cleanup succeeded. Ordinary validation errors before forwarding can be corrected without a restart. Some runtime preflight failures occur after capture replacement, so they may clear the prior trace even though the model remains reusable.

### Repeated use without needless restarts

A healthy loaded model is reused for subsequent prompts; normal queries and recoverable validation errors do not require restarting. The new CPU lifecycle tests run 64 successful queries across tiny GPT-2/Llama models, checking every retained capture bit, parameters, buffers and hooks, plus release of superseded capture objects. Separate executed-forward fault tests retain quarantine rather than hiding an unsafe failure.

A bounded real-checkpoint check loaded the pinned SmolLM2 above **once**, then completed **60 measured queries plus three warmups** in repeated “Gravity is” → “Electricity is” → “Gravity is” cycles through authenticated local HTTP. Repeat inputs produced identical output IDs and all capture bits; parameter/buffer hashes and hooks were unchanged. Weak references verified release of **11,340 superseded/final capture tensor objects**, and two early rejected requests were followed by successful queries without reload. See the [versioned record](../results/local-app-smollm2-135m-cpu-20261031/repeated_runs.json), including finite RSS observations and exact implementation/asset hashes. This is short CPU repeated-use evidence—not a promise of unlimited-session memory stability, language quality, GPU behavior, or restart-free recovery from quarantine.

```bash
# Existing exact local checkpoint and app environment; no download or overwrite:
python results/local-app-smollm2-135m-cpu-20261031/verify_repeat_runs.py --checkpoint ./models/smollm2-135m --output ./repeated-use-check.json
```

### Saved traces and privacy

Save writes a **new**, named directory under the export root, containing `tensors.safetensors`, `manifest.json` and a completion status. Existing directories are never overwritten. Exports include text prompts/token IDs, answers, image-coordinate metadata and full captured activations; treat them as potentially sensitive. The default export directory is Git-ignored. The saved-run picker lists only immediate folders under the export root. It labels apparently complete entries **unverified** and incomplete entries unavailable; listing never reads the tensor payload or proves integrity. It is a bounded convenience for a trusted local filesystem, not protection against a hostile concurrent filesystem writer. You can also paste an export path. Open saved trace verifies file inventory, shapes, byte limits and the tensor digest, then loads original values into RAM. A missing or invalid export leaves the current capture, selected values, filters and comparison visible; only a successful Open replaces them. Checksums detect changed bytes relative to the manifest; they do not authenticate a model or establish execution provenance. Model paths are labels, not cryptographic weight attestations. The checkpoint and original uploaded image are not copied into the trace export; keep them separately if needed for replay.

The server uses a fresh session token, same-origin/Host checks and no remote assets or telemetry. These are browser boundary safeguards, not protection from hostile code already running as the same local user. Do not proxy it to the internet or share its session page.

</details>

<details>
<summary>Verification</summary>

## Verification

### Integrated dependency update and UI (1.2.0)

The combined candidate uses **Python 3.11+, Transformers 5.22.1, Pillow 13.0.0 and setuptools 84.2.1**. Adapters and loading checks were updated for the new Transformers API, including Llama rotary state, T5 embedding/scaling identity, ViT/DistilBERT selectors and bounded local tokenizer assets. The local-only, safetensors and no-remote-code restrictions remain. Historical experiment records and version-pinned evidence were not rewritten as new results.

On Linux/Python 3.12.14 with CPU PyTorch 2.15.0, the integrated active suite passed **1,045 tests and 78 subtests**, with **13 prerequisite skips**. This includes real synthetic CPU model execution, capture, edits and HTTP/save/reopen flows—not a new downloaded-pretrained or GPU check. The unchanged UI contracts passed **796 mocked requests**, **850** with the public historical SmolLM2 output record, and the real-Three-math/fake-renderer contract passed. All 84 changed source/test/asset files stayed unchanged during those checks. A built **1.2.0 wheel** matched **445 packaged Python/data/asset files**; **38 additional tests** passed against the extracted wheel, with all 22 loaded package modules confirmed to come from it. Python syntax compilation and `pip check` also passed.

Fresh installed-environment and pinned-manifest `pip-audit` reports found **no known vulnerabilities among audited packages**. Both skipped the local-version `torch==2.15.0+cpu` wheel because PyPI could not resolve that build; a separate advisory query for upstream `torch==2.15.0` also found none. These Python package audits do not establish absence of unknown vulnerabilities or independently audit the vendored JavaScript/font assets. The earlier inventory listed 66 alerts against `requirements-tested.txt`; their GitHub closure has not yet been verified. The final independent security review was blocked by a tool safety denial and was not retried or bypassed, so that review gap remains.

The bare all-repository command stopped at **105 collection errors**, principally missing historical `scripts`, `tools`, experiment and example files in this sparse checkout. This is not an all-repository pass. The previous security-only checkout recorded 107 errors; different collection coverage is not evidence that historical tests were repaired. No CUDA/Windows validation or new visual-rendering evidence was added. The browser and screenshot limitations below still apply.

The **UI-only candidate, before the dependency upgrade**, passed **88 standard-library tests** (25 download, 11 authenticated loopback HTTP, 39 discovery and 13 launcher), **796 mocked UI requests**, **850** with the unchanged public SmolLM2 output record, and **22 direct layout checks**. The scene contract uses real Three.js geometry, projection and raycasting, with only the WebGL renderer replaced by a test double. It checks exact site keys, page replacement, pointer cancellation, framing at 1440 × 900 and 2576 × 1434, reduced motion, disposal and explicit fallback. These are not rendered 3D screenshots. The readout checks link existing full-model outputs to the exact saved source position; they do not run a model or project an intermediate vector into vocabulary space.

That UI-only candidate's no-install wheel matched all 18 checked app/static/vendor files, including local fonts and full licenses, and imported the curated catalog without ML packages. HTTP tests verify exact local asset bytes/MIME/security headers and rejection of unlisted paths or foreign origins. At that UI-only stage the model/runtime and tensor code were unchanged. Its full `python -m pytest -q` could not start because pytest, PyTorch and Transformers were then unavailable. No fresh model download, inference, Windows or CUDA execution was performed.

The available remote Chromium browser could not create a WebGL context (`GL_RENDERER=Disabled`). Transferring this candidate's assets to that external browser for a fallback screenshot was then blocked by an approval check; it was not retried or worked around. **This revision has not been visually rendered or accessibility-certified.** It retains the native captured-site list when graphics are unavailable. The earlier browser screenshot below belongs to the older minimal-view revision, not this new layout.

The **model-picker and download-reuse revision (2026-11-01)** passed 86 standard-library tests: 25 downloader checks, nine authenticated loopback HTTP tests, 39 discovery tests and 13 launcher tests. The HTTP tests use the real downloader with tiny synthetic files and test-only model/storage stand-ins; they do not load weights. The UI passed 568 mocked requests, 574 when replaying public historical SmolLM2 metadata, and 19 direct layout checks. The restart test confirms completed files can be browsed from a fresh app without downloading again; DOM tests check focus handoff, stale roots and failures, not rendered accessibility. A no-install wheel matched the server, downloader and three UI assets exactly; its isolated catalog import needed no ML packages. No real download, new model execution, GPU, Windows or browser rendering was tested for this feature. The full `python -m pytest -q` command could not start because pytest is absent; PyTorch and Transformers are also unavailable. The simpler view's earlier real-browser check below predates this optional picker.

The **minimal-view revision (2026-11-01)** passed 550 mocked browser requests, 556 with recorded SmolLM2 metadata, 17 layout checks, 39 discovery/loopback HTTP tests and 13 launcher tests. A no-install wheel preserved all three UI assets exactly. The full pytest command could not start because pytest, PyTorch and Transformers are absent in this environment.

The updated UI was also rendered in real remote Chromium at 1600 × 1050 using public historical SmolLM2 metadata, not a live model server. Its default rendered text fell from 425 to 138 words while all 36 displayed module buttons remained. There were no page errors. Eight real-browser checks covered keyboard-opened details, retained recorded logits/input positions, the list view, and settings Escape/focus behavior. This is one browser/fixture check—not Windows, GPU, full accessibility, raw-tensor or new inference validation. The README and this guide now lead with the everyday workflow; detailed records remain below.

The following records describe their named revisions. Historical testing limits are not a claim that later checks never happened.

```bash
python -m pytest tests/test_app_runtime.py tests/test_app_cuda.py \
  tests/test_app_selftest.py tests/test_app_storage.py \
  tests/test_app_http.py tests/test_app_install.py tests/test_app_launcher.py tests/test_app_interventions.py \
  tests/test_app_intervention_http.py tests/test_app_demo.py tests/test_app_demo_http.py \
  tests/test_app_input_preflight.py tests/test_app_setup_http.py tests/test_transformer_trace.py \
  tests/test_distilbert_trace.py tests/test_flan_t5_trace.py tests/test_vit_trace.py \
  tests/test_probes.py tests/test_interventions.py tests/test_intervention_plan.py \
  tests/test_model_ownership.py tests/test_encoder_forward.py \
  tests/test_distilbert_intervention_workflow.py tests/test_vit_trace_comparison.py \
  tests/test_vit_intervention_workflow.py tests/test_llama_trace.py \
  tests/test_app_llama.py tests/test_app_llama_storage.py tests/test_app_llama_http.py \
  tests/test_app_llama_interventions.py tests/test_app_intervention_probabilities.py \
  tests/test_app_layout.py tests/test_app_reference.py tests/test_app_reference_http.py \
  tests/test_app_repeat_runs.py tests/test_app_layer_map.py tests/test_app_layer_map_http.py -q
# Optional developer-only DOM/API contract check (Node.js; no browser install):
node tests/app_ui_contract.cjs
```

The application dependency set was verified for the layer-map snapshot on Linux, Python 3.12.14 and CPU PyTorch 2.15.0; other operating systems/interpreter combinations were not runtime-tested. The package requires Python 3.11+ to match the supported patched dependency stack.

The subsequent no-reinstall launcher change was checked separately: **45 launcher/installer/static-layout tests passed**, with **443 mocked UI requests** still passing. Launcher subprocess tests use real standard-library virtual environments and explicitly synthetic app/dependency modules, without package installation or network. They cover paths with spaces, editable-source identity, isolated imports, missing/broken environments, missing dependencies, failure exit codes and no automatic repair. This does **not** establish a fresh ML dependency installation, actual browser startup, Windows execution or GPU behavior. A bare `pytest` attempt in the newly restored, dependency-light sparse checkout stopped during collection (363 errors, chiefly unavailable PyTorch/NumPy and omitted historical scripts/tools); this is not a new all-repository pass or evidence that those tests ran.

The visual/token-readout follow-up changes only browser assets, their tests and documentation—not model execution, capture, tensor storage or the API. **47 launcher/installer/static-layout tests passed** (including 11 layout checks); the DOM contract passed **530 mocked authenticated requests**, and **536** when replaying the unchanged recorded SmolLM2 “Gravity is” metadata. That replay verifies displayed token IDs, scores and forward/input separation, not a new checkpoint inference. A no-install wheel build preserved all three UI assets byte-for-byte. The aggregate attempt still stopped with **363 collection errors** in the dependency-light sparse checkout; no ML suite, GPU execution, fresh Windows installation or actual browser rendering was completed for this UI patch. The browser available to this task could not reach the executor’s loopback app. The display uses local assets only, with no CDN or new model downloads.

The **scene-first presentation revision (2026-11-01)** is frontend-only. It passed **542 mocked authenticated requests**, **548** with the unchanged recorded SmolLM2 metadata, **14 static layout functions called directly**, and **13 standard-library launcher unit tests**. These cover the flat-view switch, settings-drawer Escape/focus, actual-site highlighting across token-specific events, stale selections, raw-value access and a retained map after a rejected Run. A no-install wheel build preserved the three UI assets exactly. The current aggregate command could not start: `python -m pytest -q` reported `No module named pytest`; PyTorch and Transformers are also absent in this restored environment. No new model inference or real HTTP integration ran for this presentation patch. Sandboxed Chromium could not launch here, and the separate browser could not reach executor localhost. At that revision, **the scene had not received actual browser rendering, mobile, screen-reader or GPU validation**; the approved design concept is not an application screenshot. No renderer sandbox was disabled and no public endpoint was opened.

The **wireframe and local-discovery follow-up (2026-11-01)** passed **39 standard-library discovery tests**, including five real loopback HTTP tests of the actual routes/classifier with explicitly synthetic model and tensor-store stand-ins; **13 launcher tests**; and **15 static layout functions**. These cover configured-root containment, current-folder detection, bounded reads/enumeration, malformed/nonfinite JSON, symlinks and nonregular files, permission errors, authentication, and preservation of the loaded-owner/capture sentinels. The DOM contract passed **549 mocked authenticated requests**, or **555** with unchanged historical SmolLM2 metadata. A no-install wheel preserved the new discovery module, server module and three UI assets exactly. The modified ML-dependent HTTP tests and full suite could not run because `pytest`, PyTorch and Transformers are absent; `python -m pytest -q` stopped with `No module named pytest`. This does not establish arbitrary-checkpoint compatibility, new model inference, actual browser rendering, Windows behavior or GPU execution. The official v1.0.0 tag is unchanged.

Tests use tiny locally generated checkpoints and synthetic tensors; they do not establish behavior for every pretrained model. The broader research suite and experimental opcode boundaries remain separate; see the [documentation guide](README.md).

The DOM contract check exercises mocked requests, not rendering. The earlier layer-map validation had nine static HTML/CSS checks; the visual/token revision’s eleven checked balanced markup, unique IDs, input/ARIA labels, native named disclosure summaries, default-closed advanced groups, visible core actions/critical notices, wrapping map cards and a programmatic (not extra tab-stop) vector heading. Disclosure regressions cover active filters, validation that reveals the relevant controls, and missing or failed comparison checks that remain visible. These are structural and simulated-interaction checks, not a browser keyboard or screen-reader certification. Real loopback HTTP and tiny local-model roundtrips were exercised separately. Chromium visual testing was blocked for those earlier checks; they provide no browser-rendering or cross-platform visual evidence.

### CUDA validation boundary

The layer-map validation executor had no visible NVIDIA device and used a CPU-only PyTorch build. CUDA policy tests use explicit synthetic API responses; they are **not GPU execution results**. Hardware-gated tests in `tests/test_app_cuda.py`, `tests/test_app_llama.py`, `tests/test_app_llama_interventions.py` and `tests/test_app_reference_http.py` exercise tiny local GPT-2/T5/DistilBERT/ViT/Llama models, same-device boundary-bit/selected-token comparisons and CPU-offload/save checks when real CUDA is available. Run them in the CUDA environment, without hiding the device:

```bash
python -m pytest tests/test_app_cuda.py tests/test_app_llama.py tests/test_app_llama_interventions.py tests/test_app_reference_http.py -q -ra
```

The focused app/adapter suite completed with **988 passed and 12 real-CUDA prerequisite skips** on the CPU executor. The UI DOM contract covered **443 mocked authenticated requests**, including optional map pagination, exact-site selection, keyboard focus targeting, stale/failed map responses, failed Run reconciliation, preserved preflight-rejected captures, demo onboarding and controlled interventions. Real loopback HTTP tests exercised all five families and saved-capture roundtrips. A freshly built wheel installed into a separate test directory passed **36 additional selected tests** for metadata-only layer maps and five-family HTTP navigation/save/reopen; import origins and all packaged UI asset bytes were verified. The Llama HTTP cases checked all 126 vectors across synthetic multi-head, grouped-query and static-Llama-3-RoPE models, including BOS-token alignment and save/reopen.

The focused suite also retains exact static-RoPE/full-logit controls, tied sharded BF16-storage conversion, demo RNG/no-write checks, installer validation/error tests, CPU encoder/ViT comparisons and intervention workflows. These checks do not establish pretrained quality or GPU behavior. Testing dependencies were restored into a fresh isolated Linux environment from official PyTorch/PyPI sources; installer tests themselves do not constitute a fresh end-user app installation. That earlier ML-validation checkout’s bare `pytest` attempt stopped with **17 collection errors**: 14 duplicate-basename import mismatches in historical experiment/study copies and three missing historical fixture/source errors. The earlier checkout recorded 47 collection errors; those counts describe different checkout coverage, not repaired historical tests. This is **not an all-repository pass**.

An additional real loopback check opened the previously verified SmolLM2 CPU export, attempted both a missing directory and a corrupt tensor digest, and verified that all 360 retained tensor hashes (1,620 vectors) and the capture revision remained unchanged. Save and a valid Open still worked afterward; this exercised retained pretrained data, not a new model execution.

A prerequisite skip is not a pass. No GPU speedup, VRAM measurement, rounding-gate fidelity, or pretrained-model quality has been established by these CPU-only checks.

### Run the standalone GPU check

From the extracted app or checkout folder, use its installed environment directly (no activation needed):

```powershell
# Windows
.\.venv\Scripts\python.exe -m looking_glass.app_selftest --device cuda:0 --max-vram-mib 1024
```

```bash
# Linux
./.venv/bin/python -m looking_glass.app_selftest --device cuda:0 --max-vram-mib 1024
```

This is an explicit, bounded diagnostic, not a background test when the app starts. It creates tiny deterministic synthetic GPT-2/T5/DistilBERT/ViT/Llama checkpoints in a temporary directory, uses only local dependencies, and never reads a user checkpoint or downloads a model. The Llama fixture uses static `llama3` RoPE. All 25 ordered family/stage checks must pass for overall success. It checks actual device placement, same-device baseline versus traced full-logit bits, selected boundary bits, CPU offload, temporary export/reopen with token/patch coordinates, repeated runs, hook cleanup and allocator-policy restoration. A failure stops further device work when ownership is uncertain rather than reporting recovery as success.

Standard output contains a JSON report with overall and per-stage **passed**, **failed**, or **skipped** states. Exit codes are **0 = passed, 1 = failed, 2 = skipped/prerequisite unavailable**. An unavailable CUDA device/build is skipped, never a CPU substitute or GPU pass. Device memory allowance covers the scoped PyTorch allocator only, with the same external-memory limitations described above. Temporary diagnostic checkpoints/traces are not your model data; the command does not persist a report unless you explicitly redirect its output.

A passing check validates only these tiny synthetic workloads on that machine and runtime. It does not establish arbitrary-checkpoint compatibility, performance improvements, rounding-gate behavior, or hostile-code isolation. The larger hardware-gated pytest checks remain available separately.

</details>
