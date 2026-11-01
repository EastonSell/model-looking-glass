# Model Looking Glass

<a id="model-looking-glass--local-model-debugger"></a>

A local **PyTorch model debugger** for mechanistic interpretability. Explore captured layer activations in your browser, inspect full vectors, compare prompts, and test temporary activation edits—without changing checkpoint files.

[Quick start](#quick-start) · [App guide](docs/LOCAL_APP.md) · [Discussions](https://github.com/EastonSell/model-looking-glass/discussions)

<a id="try-the-local-workbench"></a>

## Quick start

You need **Python 3.11+** and internet for installation. Download **Source code (zip)** from the [latest release](https://github.com/EastonSell/model-looking-glass/releases/latest), extract it, and open a terminal in the folder containing `install.py`. No Git required; this is source, not an executable.

**Windows:**

```powershell
py -3 install.py --backend cpu
```

**macOS / Linux:**

```bash
python3 install.py --backend cpu
```

Keep the terminal open. In the app, choose **Model & settings → Load tiny CPU demo**. It runs entirely in RAM, with no checkpoint download. The demo is untrained: use it to learn the controls, not to judge language quality.

Reopen with `py -3 start.py` or `python3 start.py` from the same folder—no reinstall needed. When upgrading, save your work, stop the app, and run the new release's installer. Don't copy or move an old `.venv`. [Installation help](docs/LOCAL_APP.md#install-and-start).

## Your first comparison

With the tiny demo loaded:

1. Enter `Hi!` and click **Run**.
2. In **Layers**, switch to **List**. Select the block `transformer.h.0`, then choose position **2** (`!`) in **Token / position**.
3. Open **Reference → Keep selected vector**. Run `Hi?` without reloading the model.
4. Select the same block and position **2** (`?`), then **Reference → Compare selected vector**.

You'll see both vectors and their feature-by-feature differences. This compares recorded values, not word meanings. **Exact values** keeps the original numbers and raw bits available. The reference is RAM-only and isn't included in **Save capture**.

<a id="what-to-use"></a>
<a id="try-your-own-model"></a>

## Bring a trained model

Open **Model & settings**, browse to a checkpoint folder, choose **Inspect folder**, then **Load model**. Or use **Get a model** for a small, pinned Hugging Face catalog; nothing downloads until you choose **Download**.

Compatible **GPT-2, standard Llama, DistilBERT, T5 and ViT safetensors** exports are supported—not every checkpoint in those families. [Exact compatibility](docs/LOCAL_APP.md#checkpoint-compatibility).

- **Explore:** select captured sites and token/image positions; inspect full activation vectors.
- **Compare:** keep one reference across prompts on the same loaded GPT-2 or supported Llama model.
- **Experiment:** on GPT-2 or supported Llama, temporarily zero a vector or change a feature, with baseline/control comparisons on a fixed input.

Recorded token predictions come from the whole model, not from decoding a hidden vector. The layer view shows captured sites, not inferred reasoning or causal connections.

## Before you run

**Save captures you want to keep.** An accepted new run replaces the previous RAM capture, even if execution later fails. Exports may contain private prompts and activations. Keep this single-user app local; don't expose its server online.

CPU is enough to start. CUDA paths exist, but actual GPU validation is still outstanding. Memory estimates and limits aren't fit guarantees. [Devices, memory and privacy](docs/LOCAL_APP.md#memory-failures-and-privacy).

<a id="what-the-research-supports"></a>
<a id="smaller-checkout-intact-evidence"></a>
<a id="go-further"></a>
<a id="contribute"></a>

## Learn more & contribute

[App guide](docs/LOCAL_APP.md) · [Python tutorial](docs/GPT2_TRACE_WORKFLOW.md) · [Code map](looking_glass/README.md) · [API & research index](docs/README.md)

[Findings and limits](docs/PROJECT_STATUS.md) · [Roadmap](docs/RESEARCH_ROADMAP.md) · [Results](RESULTS.md) · [Historical records](archives/README.md)

[Contribute](CONTRIBUTING.md), [discuss ideas](https://github.com/EastonSell/model-looking-glass/discussions), or [report a vulnerability privately](SECURITY.md). See our [code of conduct](CODE_OF_CONDUCT.md) and [accessibility guidance](ACCESSIBILITY.md).

This guide describes main; older releases may differ. Licensed under [MIT](LICENSE), with [upstream licenses](looking_glass/app_static/vendor/README.md) retained for bundled assets.
