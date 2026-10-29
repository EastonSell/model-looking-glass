# Model Looking Glass

**Tools for observing a neural network, testing interventions, and measuring what actually changed.** The research goal is a causal debugger—not a claim that arbitrary model behavior is already explained or safely repairable.

Start with **direct activation capture**, explicit execution coordinates, and controlled intervention comparisons. The auxiliary-gradient reporting channel is an experimental control: it adds no information to an activation already available to copy, costs more in the original CPU study, and can lose subnormal bits.

## What to use

| Task | Entry point | Scope |
|---|---|---|
| Observe named boundaries | [`TraceSession`, `TapPoint`](looking_glass/probes.py) | Local, single-threaded, bounded snapshots; no implicit tracing of every neuron |
| Capture, inject, or patch | [Intervention primitives](docs/INTERVENTION_PLANS.md) | Explicit sites and controls, exclusive model ownership, cleanup checks |
| Record and compare traces | [Schema](docs/TRACE_SCHEMA.md), [store](looking_glass/trace_store.py), [localization](docs/LOCALIZATION.md) | Execution metadata and first observed differences, not inferred reasoning |
| Evaluate changes | [Causal metrics](docs/CAUSAL_METRICS.md), [behavioral evaluation](docs/BEHAVIORAL_EDIT_EVALUATION.md) | Paired outcomes and locality observations; no broad capability guarantee |
| Work with transformers | [GPT-2](docs/TRANSFORMER_TRACE.md), [DistilBERT](docs/DISTILBERT_TRACE.md), [T5](docs/FLAN_T5_TRACE.md), [ViT](docs/VIT_TRACE.md) | Documented eager-model adapters, not universal architecture support |

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

The complete historical suite has additional dependencies and archived fixtures. Do not treat a focused pass, missing-dependency skip, or unrun test as a full-suite pass. See [this cleanup's validation](docs/MAINTENANCE_2026-10-29.md).

## What the research supports

- **Observation baseline:** direct and gradient channels agreed under the recorded floating-point checks; gradient capture took 2.60–3.81× direct latency (across-seed median ratios) in a small CPU model. Dense taps narrowed where an injected difference first appeared. These are not LLM-scale or bitwise-fidelity results. [Original results](RESULTS.md)
- **Important counterexample:** auxiliary-gradient readout can lose subnormal bits even when `torch.equal` reports equality. Prefer direct capture; use raw-bit checks when bitwise fidelity matters. [Denormal control](docs/GRADIENT_DENORMAL_V1.md)
- **Interventions are not explanations by themselves:** small language/vision studies include positive score shifts, failed specificity controls, ineligible baselines, and locality failures. No general model-repair success is established. [Research evidence guide](docs/PROJECT_STATUS.md#research-evidence)

All outcomes—including failures and negative controls—remain accessible. The [full pre-cleanup research index](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/README.md) is preserved at an immutable commit; it is historical, not a current support matrix.

## Smaller checkout, intact evidence

Large historical tensor/archive payloads are indexed in [`archives/results-manifest.json`](archives/results-manifest.json), with their original paths, Git blob hashes, sizes, and source commit. Uncompressed reports and decision records remain in the current tree; compressed records are included in the archive manifest. [Archive and restore instructions](archives/README.md) explain selective restoration and integrity verification.

[Ten duplicate historical auditor entry points](archives/SOURCE_CONSOLIDATION.md) were also consolidated into their retained, byte-identical result-local copies.

This reduces the **current checkout**, not Git history. No history is rewritten; a full clone still contains old objects. For a small new checkout:

```bash
git clone --filter=blob:none --depth=1 https://github.com/EastonSell/model-looking-glass.git
```

[Roadmap](docs/RESEARCH_ROADMAP.md) · [Original protocol](PROTOCOL.md) · [Mathematical limits](docs/MATHEMATICS.md)
