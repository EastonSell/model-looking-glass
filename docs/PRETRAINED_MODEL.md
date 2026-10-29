# Pinned pretrained CPU model

This loader supplies a real, small pretrained transformer for observation
experiments. It is an input to research, not evidence that a trace explains a
model's behavior or that a particular intervention is causal. The existing
synthetic results and protocol remain separate.

## Identity and sources

Both model and tokenizer come from
[`distilbert/distilgpt2` at commit `2290a62682d06624634c1f46a6ad5be0f47f38aa`](https://huggingface.co/distilbert/distilgpt2/tree/2290a62682d06624634c1f46a6ad5be0f47f38aa).
The [model card](https://huggingface.co/distilbert/distilgpt2/blob/2290a62682d06624634c1f46a6ad5be0f47f38aa/README.md)
describes DistilGPT2 as a pretrained distilled GPT-2 model and identifies its
license as Apache-2.0. Its training data, limitations and risks are described
in that card. The pinned [safetensors file](https://huggingface.co/distilbert/distilgpt2/blob/2290a62682d06624634c1f46a6ad5be0f47f38aa/model.safetensors)
is 352,824,413 bytes; its published LFS SHA-256 is in the table below.

Each file actually supplied to Transformers is copied into a temporary
allowlisted directory inside the requested cache, while its size and SHA-256
are checked. Neighboring unverified files are not included. The exact hashes
are also recorded in the returned `metadata["asset_sha256"]` mapping.

| File | SHA-256 |
| --- | --- |
| `config.json` | `4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a` |
| `generation_config.json` | `b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805` |
| `tokenizer.json` | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `tokenizer_config.json` | `5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f` |
| `vocab.json` | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| `merges.txt` | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| `model.safetensors` | `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b` |

The metadata describes the files verified at load time. It is not a hash of
the live model after a caller changes its parameters, configuration or mode,
and it is not a defense against a hostile process modifying a staging file
during loading.

## Usage

Install the repository's PyTorch dependency and the optional
`transformers==4.57.3` package. The initial CPU verification used Python
3.11.15, PyTorch 2.8.0+cu128 and Transformers 4.57.3 on Windows; it did not use
CUDA. Other combinations have not been verified for this loader. The library
never installs packages automatically.

Give the loader an explicit, writable cache directory. It is offline by
default; the following opts in to downloading the pinned public files when
they are absent. Set `HF_HOME` to a workspace directory if you also want to
keep Hugging Face's other caches there. The download needs about 353 MB for
weights and staging needs approximately the same additional free disk space.

```python
from pathlib import Path
import torch
from looking_glass.pretrained import load_distilgpt2

bundle = load_distilgpt2(cache_dir=Path("work/hf-cache/hub"), allow_download=True)
inputs = bundle.encode("The sky is blue.", max_tokens=32)
with torch.inference_mode():
    output = bundle.model(**inputs)
print(output.logits.shape)
print(bundle.metadata)
```

Subsequent calls can omit `allow_download`; all required pinned files must be
available in the specified cache. `CheckpointUnavailableError` identifies an
offline cache miss or download failure and chains the underlying exception;
`CheckpointIntegrityError` rejects wrong sizes or hashes. The loader rejects
missing, unexpected or mismatched learned weights and loading errors reported
by Transformers. It uses `use_safetensors=True`, `trust_remote_code=False`,
and `local_files_only=True` when loading the staged model and tokenizer.

The returned model is initially in evaluation mode with float32 CPU parameters,
eager attention, and generation cache disabled. The separate GPU execution
service, not this loader, must coordinate GPU work. The returned model is a
normal mutable Transformers model; callers can bypass the encoder or change
its state and are responsible for their own execution limits and manifests.

`bundle.encode` accepts one nonempty string of at most 8,192 characters and
between 1 and 128 tokens, controlled by `max_tokens`. Overlong inputs are
rejected rather than silently shortened. It returns batch-one `input_ids` and
`attention_mask` tensors on CPU; it does not save prompts or activations.

## Tests and reproduction

The ordinary unit suite includes an offline, randomly initialized tiny GPT-2
fixture to check safe staging, exclusion of unverified neighboring files,
rejection of missing learned weights, and bounded encoding. It is a loader
regression fixture, not evidence about the pretrained checkpoint.

An additional opt-in smoke test loads and hashes the pinned checkpoint,
checks known token IDs and runs a CPU forward after staging cleanup. For
PowerShell, set the cache path explicitly and run:

```powershell
$env:HF_HOME = (Resolve-Path 'work/hf-cache').Path
$env:LOOKING_GLASS_TEST_CACHE = (Join-Path $env:HF_HOME 'hub')
$env:OMP_NUM_THREADS = '1'
python -m unittest discover -s tests -p test_pretrained.py -v
```

The smoke test is not a latency, throughput, memory or causal evaluation.
