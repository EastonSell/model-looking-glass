# Model Looking Glass

**Tools for observing a neural network, testing interventions, and measuring what actually changed.** The research goal is a causal debugger—not a claim that arbitrary model behavior is already explained or safely repairable.

Start with **direct activation capture**, explicit execution coordinates, and controlled intervention comparisons. The auxiliary-gradient reporting channel is an experimental control: it adds no information to an activation already available to copy, costs more in the original CPU study, and can lose subnormal bits.

**[Documentation: start here](docs/README.md)** · [Findings and limits](docs/PROJECT_STATUS.md#research-evidence) · [One living roadmap](docs/RESEARCH_ROADMAP.md)

## Try the local workbench

```bash
python install.py
```

With Python 3.11+, from a trusted clone, this creates an isolated environment and opens a local settings page. Linux/Windows default to the official CUDA 13.2 PyTorch build (requires a compatible NVIDIA GPU/driver); use `python install.py --backend cpu` without CUDA. The installer never installs a driver. Browse an explicitly chosen model folder or paste a compatible **local safetensors checkpoint** path, enter a prompt (or a ViT image), and capture every supported boundary or choose specific hooks. Full vectors stay **in RAM by default**; navigate recorded token/patch positions and raw values/bits, optionally view approximate grouped coordinates, search across all captured events, and explicitly save/reopen lossless traces from the saved-run picker. Unload the model without losing the capture. A live GPT-2 capture also supports controlled fixed-input activation experiments, kept separate from the original trace. Hooks do not alter checkpoint files.

The workbench has a **CUDA-first paths for GPT-2, T5, DistilBERT and ViT**, with CPU fallback. Choose Auto or an explicit device in Settings. CUDA implementation is hardware-gated and has **not been executed on a GPU in the current validation environment**. This is not arbitrary-checkpoint or every-operation support. Grouped views are not semantic explanations. Validate a GPU locally with `python -m looking_glass.app_selftest`; it reports passed/failed/skipped checks using tiny synthetic models, without downloads or your checkpoints. [Installation, compatibility, memory limits and privacy](docs/LOCAL_APP.md).

**Inspect folder** checks local input assets before loading weights and keeps inspection separate from the loaded model. Missing tokenizer/processor files are diagnosed before replacing your live model; a failed load leaves the retained capture readable.

**No checkpoint yet?** Choose **Load tiny demo**, then Run. This builds an untrained miniature GPT-2 entirely in CPU RAM—no download or checkpoint files—so you can learn capture navigation and controlled edits immediately. Its random output is not useful language or GPU validation.

## What to use

| Task | Entry point | Scope |
|---|---|---|
| Observe named boundaries | [`TraceSession`, `TapPoint`](looking_glass/probes.py) | Local, single-threaded, bounded snapshots; no implicit tracing of every neuron |
| Capture, inject, or patch | [Intervention primitives](docs/INTERVENTION_PLANS.md) | Explicit sites and controls, exclusive model ownership, cleanup checks |
| Record and compare traces | [Schema](docs/TRACE_SCHEMA.md), [store](looking_glass/trace_store.py), [localization](docs/LOCALIZATION.md) | Execution metadata and first observed differences, not inferred reasoning |
| Evaluate changes | [Causal metrics](docs/CAUSAL_METRICS.md), [behavioral evaluation](docs/BEHAVIORAL_EDIT_EVALUATION.md) | Paired outcomes and locality observations; no broad capability guarantee |
| Work with transformers | [GPT-2 tutorial](docs/GPT2_TRACE_WORKFLOW.md), [DistilBERT](docs/DISTILBERT_TRACE.md), [T5](docs/FLAN_T5_TRACE.md), [ViT](docs/VIT_TRACE.md) | Documented eager-model adapters, not universal architecture support |

See [project scope](docs/PROJECT_STATUS.md) for the boundary between reusable components, experimental infrastructure, and historical study runners. Retained code is not automatically a supported public API. The shared-service and GPU scheduling prototypes are **not** a validated general GPU service.

## Quick start

Python 3.10+ is declared; choose PyTorch using its [official installation instructions](https://pytorch.org/get-started/locally/). The optional `transformer` extra pins Transformers 4.57.3. Dependency ranges are not a claim that every permitted combination has been tested.

After cloning, enter the repository (`cd model-looking-glass`). Install the package into your chosen environment with `python -m pip install -e .` after selecting its PyTorch build. Use `python -m pip install -e ".[transformer,test]"` only when you need the optional adapter/test dependencies.

```python
import torch
from looking_glass.model import TinyResidual
from looking_glass.probes import TraceSession

model = TinyResidual(width=16)
with TraceSession(model, mode="direct", max_events=64,
                  max_tensor_bytes=1_000_000) as trace:
    output = model(torch.randn(8, 16), route=0)
print([event.metadata() for event in trace.events])
```

Captured activations may contain private information. Limits count logical recorded bytes, not peak RAM/VRAM. No model or dataset is downloaded by this example or the original experiment. Pretrained loaders have separate explicit download/cache requirements.

```bash
# Core observation checks (requires PyTorch; no checkpoint download):
python -m unittest discover -s tests -p 'test_probes.py' -v
python -m unittest discover -s tests -p 'test_interventions.py' -v
# Original synthetic experiment; never overwrites an existing directory:
python -m looking_glass.experiment --output results/local/my-run
# Archive tooling checks (stdlib + Git only):
python -m unittest discover -s tests -p 'test_archive_results.py' -v
```

The complete historical suite has additional dependencies and archived fixtures. The default `python -m pytest tests` run explicitly excludes 57 historical opcode-exception experiments: Python documents that their injection mechanism can cause undefined interpreter behavior. Their exact cases and assertions are preserved behind explicit, per-case process-isolated opt-in. See [test boundaries and controls](docs/EXPERIMENTAL_OPCODE_TESTS.md).

Do not treat a focused pass, prerequisite skip, excluded experiment, or unrun test as validated behavior. See [this cleanup's actual results](docs/MAINTENANCE_2026-10-29.md).

## What the research supports

- **Observation baseline:** direct and gradient channels agreed under the recorded floating-point checks; gradient capture took 2.60–3.81× direct latency (across-seed median ratios) in a small CPU model. Dense taps narrowed where an injected difference first appeared. These are not LLM-scale or bitwise-fidelity results. [Original results](RESULTS.md)
- **Important counterexample:** auxiliary-gradient readout can lose subnormal bits even when `torch.equal` reports equality. Prefer direct capture; use raw-bit checks when bitwise fidelity matters. [Denormal control](docs/GRADIENT_DENORMAL_V1.md)
- **Interventions are not explanations by themselves:** small language/vision studies include positive score shifts, failed specificity controls, ineligible baselines, and locality failures. No general model-repair success is established. [Research evidence guide](docs/PROJECT_STATUS.md#research-evidence)

All outcomes—including failures and negative controls—remain accessible. The [full pre-cleanup research index](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/README.md) is preserved at an immutable commit; it is historical, not a current support matrix.

## Smaller checkout, intact evidence

Large historical tensor/archive payloads are indexed in [`archives/results-manifest.json`](archives/results-manifest.json), with their original paths, Git blob hashes, sizes, and source commit. Uncompressed reports and decision records remain in the current tree; compressed records are included in the archive manifest. [Archive and restore instructions](archives/README.md) explain selective restoration and integrity verification.

[Ten duplicate historical auditors and a superseded three-script preparation workflow](archives/SOURCE_CONSOLIDATION.md) were removed from active entry points, with canonical copies or immutable historical source links retained.

The [documentation consolidation](archives/README.md#historical-documentation) also replaces scattered historical manuals with a short reading path, one roadmap, a findings map and an experimental infrastructure reference.

This reduces the **current checkout**, not Git history. No history is rewritten; a full clone still contains old objects. For a small new checkout:

```bash
git clone --filter=blob:none --depth=1 https://github.com/EastonSell/model-looking-glass.git
```

[Documentation](docs/README.md) · [Roadmap](docs/RESEARCH_ROADMAP.md) · [Original protocol](PROTOCOL.md) · [Mathematical limits](docs/MATHEMATICS.md)
