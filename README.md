# Model Looking Glass

A local PyTorch debugger for mechanistic interpretability. See what each layer would say next, inspect full activation vectors, compare prompts, and try temporary edits—without changing checkpoint files.

![Model Looking Glass lens view: a grid of GPT-2's top next-token guess at every layer and input position, with the token " Paris" tracked from rank 6,060 at layer 0 to rank 1 at layer 10.](docs/images/model-looking-glass-lens.png)

*GPT-2 small on CPU, prompt "The Eiffel Tower is located in the city of". Rows are layers, columns are input tokens; each cell is that layer's top next-token guess, and colour is the rank of " Paris".*

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

After setup, launch with `python start.py` (or `py -3 start.py` on Windows). No model-size or GPU flags are needed. The automatic model preflight uses half of currently available host RAM, up to 16 GiB; automatic CUDA allowance leaves 10% VRAM or 256 MiB of headroom, whichever is larger. Explicit caps remain available for advanced use. These are storage/admission estimates, not guarantees of peak memory use.

1. Open **Model & settings** and choose your local checkpoint folder.
2. Leave device, mode and GPU allowance on **Auto**, then **Inspect folder → Load model**. Available CUDA is preferred without launch flags. GPT-2 and supported Llama open in **Chat**; other families use **Capture**.
3. In Chat, send messages without activation capture. Cached generation uses the model’s tokenizer chat template, up to 512 new tokens by default (less for small contexts), within the model and memory limits. Leave Chat’s context field blank for automatic sizing that can shrink to fit GPU memory; an explicit number is a strict request. Without a chat template, only one raw prompt is supported; choose **New chat** for another.
4. For inspection, select **Capture** under **Mode for next load**, then load again. Enter text or choose an image and **Run & capture**. Select a layer and token/patch for **Exact values**. **Reference / Edit** work on GPT-2 and supported Llama.

Chat history stays in the browser tab’s RAM and clears on reload or model replacement. The response reports context limits and any older messages omitted; context is not unlimited. CUDA Chat may use BF16/FP16 to reduce memory, with the actual dtype displayed. Outputs can differ from float32. Capture retains its float32 numerical contract. Changing modes reloads the model, clears live chat/capture state, and leaves saved files intact.

For a live GPT-2 or supported Llama capture, **Experiment** can sweep selected layer outputs, repeat tests on several prompts, and patch an explicitly selected activation between prompts. [Experiment guide](docs/LOCAL_APP.md#run-an-automated-component-sensitivity-experiment).

Compatible GPT-2, standard Llama, DistilBERT, T5 and ViT **safetensors** exports work—not every checkpoint. [Supported formats](docs/LOCAL_APP.md#checkpoint-compatibility). **Get a model** offers optional Hugging Face downloads; **Load tiny CPU demo** is an optional untrained sandbox.

Reopen with `py -3 start.py` on Windows or `python3 start.py` on Linux/macOS. Keep the folder in place and terminal open.

<a id="your-first-comparison"></a>

## The lens view

After **Run & capture** on GPT-2 or a supported Llama, the centre of the app is a **layer × token grid**. Each cell answers one question: *if the model stopped at this layer, which token would it say next?* It is a [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens): the captured block output is passed through the model's own final norm and unembedding.

- **Track a token** (type it, or click one in the read-out) to colour the grid by that token's rank and plot its rank through the layers.
- **Click a cell** for its top tokens, then **Open exact vector** to inspect, edit or patch that same activation.
- **Switch to Field** for the same read-out as a moving field of text, with a depth control that scrubs through the layers.
- **Switch the read-out method to Jacobian lens** to carry each layer into the final layer's basis before decoding. It needs a lens for the model: fit one on an NVIDIA GPU, or import a published one.
- The last layer's read-out must equal the token the model really produced; the app checks this on every read-out and says so.

Earlier layers were never trained to be read this way, so lower rows are an approximation—useful for seeing *when* an answer appears, not proof of *how*. [Lens guide](docs/LOCAL_APP.md#lens-what-each-layer-would-say-next).

## How this relates to other tools

This is a local app with a UI, not a research library. For programmatic work at scale, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [nnsight](https://github.com/ndif-team/nnsight) are the mature choices for hooks, patching and lenses across many architectures. Model Looking Glass adds a no-code workbench around the same basic methods—capture, logit lens, zero/add/replace edits and cross-prompt patching—with exact-value inspection and explicit controls. It also implements the Jacobian lens from Anthropic's 2026 global-workspace paper, fitted locally on a GPU or imported from a published file. It does **not** implement that paper's J-space analysis, sparse autoencoders or feature attribution.

## Before you run

**Save captures before replacing them.** Keep the app local; exports can contain private prompts. Memory estimates aren't guarantees. In the result panel, token predictions are whole-model output; only the lens view decodes hidden vectors, and it labels itself as an approximation below the last layer.

<a id="what-the-research-supports"></a>
<a id="smaller-checkout-intact-evidence"></a>
<a id="go-further"></a>
<a id="contribute"></a>
<a id="learn-more--contribute"></a>

[Contribute](CONTRIBUTING.md) · [Code map](looking_glass/README.md) · [Docs](docs/README.md) · [Research archive](docs/README.md#research-archive) · [Security](SECURITY.md) · [MIT license](LICENSE)
