# Model Looking Glass

A local PyTorch model debugger. Explore captured layers, inspect full activation vectors, compare prompts, and try temporary edits—without changing checkpoint files.

**[Download the app](https://github.com/EastonSell/model-looking-glass/releases/latest)** · [App guide](docs/LOCAL_APP.md) · [Discussions](https://github.com/EastonSell/model-looking-glass/discussions)

<a id="model-looking-glass--local-model-debugger"></a>
<a id="try-the-local-workbench"></a>

## Quick start

Download **model-looking-glass-app.zip** from the release's **Assets**, extract it, and open a terminal there. Requires **Python 3.11+**. Choose your setup:

| | NVIDIA GPU | CPU |
|---|---|---|
| Windows | `py -3 install.py --backend cuda` | `py -3 install.py --backend cpu` |
| Linux | `python3 install.py --backend cuda` | `python3 install.py --backend cpu` |
| macOS | — | `python3 install.py --backend cpu` |

CUDA needs a compatible NVIDIA GPU and driver; drivers aren't installed for you. Dependencies and checkpoints download separately. [GPU setup](docs/LOCAL_APP.md#device-selection-and-gpu-semantics).

<a id="what-to-use"></a>
<a id="try-your-own-model"></a>
<a id="bring-a-trained-model"></a>

## Open your model

1. Open **Model & settings** and choose your local checkpoint folder.
2. Choose **Auto / CUDA / CPU**, then **Inspect folder → Load model**.
3. Enter text or choose an image, then **Run & capture**. Select a layer and token/patch for **Exact values**. **Reference / Edit** work on GPT-2 and supported Llama.

Compatible GPT-2, standard Llama, DistilBERT, T5 and ViT **safetensors** exports work—not every checkpoint. [Supported formats](docs/LOCAL_APP.md#checkpoint-compatibility). **Get a model** offers optional Hugging Face downloads; **Load tiny CPU demo** is an optional untrained sandbox.

Reopen with `py -3 start.py` on Windows or `python3 start.py` on Linux/macOS. Keep the folder in place and terminal open.

<a id="your-first-comparison"></a>

## Before you run

**Save captures before replacing them.** Keep the app local; exports can contain private prompts. Memory estimates aren't guarantees. Token predictions are whole-model output, not words decoded from hidden vectors.

<a id="what-the-research-supports"></a>
<a id="smaller-checkout-intact-evidence"></a>
<a id="go-further"></a>
<a id="contribute"></a>
<a id="learn-more--contribute"></a>

[Contribute](CONTRIBUTING.md) · [Code map](looking_glass/README.md) · [Research & API](docs/README.md) · [Security](SECURITY.md) · [MIT license](LICENSE)
