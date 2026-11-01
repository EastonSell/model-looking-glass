# Model Looking Glass

<a id="model-looking-glass--local-model-debugger"></a>

Run a local model and look inside its layers. The browser workbench lets you inspect captured activations, compare vectors across prompts, and try temporary activation edits without changing checkpoint files.

<a id="try-the-local-workbench"></a>

## Quick start

You need **Python 3.11+** and internet for the initial installation. Download the [source ZIP from main](https://github.com/EastonSell/model-looking-glass/archive/refs/heads/main.zip), extract it, and open a terminal in the folder containing `install.py`. This is Python source, not a standalone executable.

**Windows:**

```powershell
py -3 install.py --backend cpu
```

**macOS or Linux:**

```bash
python3 install.py --backend cpu
```

Keep the terminal open. In the app, open **Model & settings**, choose **Load tiny CPU demo**, then **Run**. Select a layer and a token to inspect its captured vector. The demo needs no model download; it is untrained, so don't expect useful answers.

To open the app again, run `py -3 start.py` on Windows or `python3 start.py` on macOS/Linux. Keep the extracted folder in place. You don't need to reinstall. [Installation help and older releases](docs/LOCAL_APP.md#install-and-start).

<a id="what-to-use"></a>

## Try your own model

In **Model & settings**, browse to a local checkpoint folder, inspect it, then load it. Supported formats include compatible GPT-2, standard Llama, DistilBERT, T5 and ViT **safetensors** exports—not every model bearing those names. Need a checkpoint? Open **Get a model** for a small, pinned Hugging Face catalog. Downloads happen only when you choose **Download**; memory estimates are not fit guarantees. [Compatibility details](docs/LOCAL_APP.md#checkpoint-compatibility).

- **Explore:** select a captured layer and input position; open exact values when you need them.
- **Compare:** keep a reference vector, run another prompt, and inspect the differences.
- **Experiment:** with a fresh GPT-2 or supported Llama capture, temporarily change a vector or feature and compare the model's output.

The layer view shows recorded data, not a diagram of the model's reasoning. A changed activation does not by itself explain a behavior.

## Before you run

**Save anything you want to keep.** An accepted new run replaces the previous RAM capture, even if execution later fails. Saved traces can contain private prompts and activations. Keep this single-user app local; don't expose its server online.

CPU is enough to begin. CUDA paths exist, but their hardware tests have not run in the recorded validation environment. Large captures need extra memory; configured limits don't guarantee that a model will fit. [Devices, memory and privacy](docs/LOCAL_APP.md#memory-failures-and-privacy).

<a id="what-the-research-supports"></a>
<a id="smaller-checkout-intact-evidence"></a>

## Go further

- [App guide](docs/LOCAL_APP.md): loading, exploring, comparing and saving.
- [Python library tutorial](docs/GPT2_TRACE_WORKFLOW.md) and [API references](docs/README.md): use the tools in your own code.
- [Research findings and limits](docs/PROJECT_STATUS.md), [roadmap](docs/RESEARCH_ROADMAP.md), and [original results](RESULTS.md).
- [Historical records and restoration](archives/README.md): original evidence remains available, including failed experiments.

This guide describes main. Existing release tags, including v1.0.0, are unchanged and may lack newer controls.
