# Model Looking Glass

**Mission: build a causal debugger for neural networks that can discover how a behavior is produced, test that explanation, and help change the behavior precisely.**

The long-term goal is to turn internal observations into explanations that predict interventions on unseen inputs, then use that understanding for targeted model improvements. Language, vision, multimodal systems, training and inference are all within the research direction. The current small experiment is the first instrument, not the ceiling.

- [Research roadmap: observation → tracing → tested explanations → targeted changes](docs/RESEARCH_ROADMAP.md)
- [Efficiency plan: parallel agent teams and a shared GPU experiment service](docs/EFFICIENCY_PLAN.md)

The roadmap is a research plan, not a claim that its milestones are complete. The original `TraceSession` remains single-threaded. Separate components now provide bounded trace snapshots, trace metadata, experiment records, localization, causal metrics, and GPU scheduling primitives; they do not yet constitute a validated shared GPU service. No pretrained-model study, tested causal explanation, or model repair result is claimed by the initial experiment.

## Current building blocks

- The [trace schema](docs/TRACE_SCHEMA.md) records execution coordinates separately from tensor values and hypotheses. Unknown coordinates remain unknown.
- The [GPT-2 trace adapter](docs/TRANSFORMER_TRACE.md) captures selected layer, module, local token and invocation coordinates; generation and cache positions remain unknown unless established separately. The [ordered intervention plan](docs/INTERVENTION_PLANS.md) can capture, inject and patch at specified boundaries.
- The [fair queue](docs/FAIR_QUEUE.md) rotates among admitted teams; it does not by itself guarantee admission fairness or equal GPU time.
- The [memory budget](docs/MEMORY_BUDGET.md) and [resource profiles](docs/RESOURCE_PROFILE.md) track conservative leases, trace allowances and workload-specific measurements. Estimates and logical byte counts are not peak-memory guarantees or proof of device enforcement; no shared-service GPU workload profile has yet been approved.
- The [resident worker](docs/RESIDENT_WORKER.md) and [broker](docs/BROKER.md) reuse one trusted model for queued forwards. Their current execution is serial; GPU-wide admission and concurrent device throughput require separate integration and validation.
- The [local CPU service](docs/SERVICE.md) exposes the pinned DistilGPT2 model to authenticated clients, with a two-token input limit and serial execution. A separate [microbatch primitive](docs/MICROBATCH.md) and [sampled batch validator](docs/BATCH_VALIDATION.md) are not yet integrated into that service.
- The [device supervisor](docs/SUPERVISOR.md) coordinates a cooperative owner process, a device lease, and a sticky quarantine marker. It has CPU tests, but has not yet been integrated and validated as a shared GPU service.
- The [local transport](docs/IPC.md) lets authenticated cooperating client processes submit bounded forwards over loopback. It does not by itself enforce GPU budgets or provide trace delivery.
- The [experimental fair IPC bridge](docs/IPC_FAIR_BRIDGE.md) joins bounded forward-only admission and receipt custody in CPU tests. GPU service composition requires separate admission and supervision.
- The [trace delivery prototype](docs/TRACE_TRANSPORT.md) tests bounded, single-event custody on CPU. The service does not yet send traces through IPC or provide a complete multi-event publication protocol.
- A [CPU trace owner](docs/CPU_TRACE_OWNER.md) composes the [traced worker](docs/TRACED_WORKER.md), [producer envelope](docs/TRACE_PRODUCER.md) and [host custody ledger](docs/HOST_BUDGET.md) for private in-process jobs. They do not yet form a live broker-to-IPC trace path or a physical memory limit.
- The [experiment registry](docs/REGISTRY.md) records manifests, artifacts and terminal states. [Research cases](docs/RESEARCH_CASES.md), [localization](docs/LOCALIZATION.md), and [causal metrics](docs/CAUSAL_METRICS.md) support controlled evaluations; the public held-out cases are not a protected blind benchmark. [Transformer trace comparison](docs/TRANSFORMER_LOCALIZATION.md) requires independently verified pairing and execution context.

These components have their own tests and limits; their presence alone does not establish a validated shared GPU service, causal discovery, or a general throughput claim.

## First experiment

Can many transparent observation points help us follow a model's internal computation, and does reporting a vector through a separate gradient offer anything over copying it directly?

This repository compares two reporting channels at the **same named observation points**, plus an ID-only baseline. It also separates a vector's contents from its execution metadata and tests how sparse versus dense observation changes where an introduced numerical difference first becomes visible.

**Initial result:** the observation API works on the tested synthetic model, but the gradient reporting channel adds **no vector information** over direct capture and takes **2.60–3.81× its latency** (median paired ratios across seeds) in this small CPU experiment. Dense taps narrow the first-observed difference to the injected tap; sparse taps report a later block output. All 48 training-transparency comparisons passed the recorded floating-point checks, and all 26 unit tests passed. The run did not record denormal mode or preserve tensor bits, so those checks do not establish bitwise equality. See [RESULTS.md](RESULTS.md) for measurements, raw evidence and limitations—not an LLM-scale claim.

## The three channels

- **Direct:** a forward hook saves `activation.detach().clone()` and returns `None`, leaving the original output alone.
- **Gradient:** the hook creates a separate zero-valued leaf `q` and calculates `grad_q sum(stopgrad(activation) * q)`. For finite inputs the derivative is the activation. `q` is ephemeral, outside the model and every optimizer.
- **ID-only:** log node labels and invocation counts without saving tensor values.

All channels use genuine identity taps. There is no floating-point cancellation gate or hidden payload. The gradient channel already reads the activation to form its reporting objective; it does not gain information unavailable to direct capture. Its reliability and overhead are the experiment, not an assumed benefit.

The identity is mathematical; a [CPU denormal control](docs/GRADIENT_DENORMAL_V1.md) shows that floating-point gradient readout can lose subnormal bits while `torch.equal` still reports equality. Use raw-bit checks when bitwise fidelity matters.

## What the experiment checks

1. Unchanged task outputs, scalar loss, parameter gradients, input gradients and one SGD update.
2. Equality of direct and gradient recordings at matched locations.
3. An identical-output branch control: equal vectors cannot identify which branch executed, while explicit node metadata can.
4. Where a fixed numerical perturbation is first observed with sparse versus dense taps.
5. Complete per-call runtime, event counts and exact bytes of saved tensor values. Saved bytes are **not peak memory**.

The initial model is a small randomly initialized residual network with a controlled route. It is not a pretrained language model. Trace labels identify instrumented boundaries, not a unique last neuron or the model's reasoning.

## Exploratory intervention study

The [synthetic causal v1 study](docs/CAUSAL_RESULTS_V1.md) records fixed predictions and tests clean-state patches on 16 public held-out inputs. Its selected injection-site patch and the single random draw, a final-state patch, produced the same measured scalar effects, illustrating why restoration alone does not identify a unique cause. One held-out vector was previously used during development, so the result is explicitly exploratory; the raw run and protocol clarification are preserved alongside the report.

The [synthetic v2 study](results/synthetic-causal-v2-20261029/README.md) records 57,408 forwards across six fixed small models and 54 model/injection cells. First-changed blocks matched the injected block in 54/54 cells, while a magnitude-based rule matched 33/54. The known feedforward structure favors the first-change rule; a successful patch does not uniquely identify an operation. The saved-data audit does not independently verify execution, prediction chronology or exposure, so the result remains unconfirmed. The [resource clarification](docs/CAUSAL_CONFIRMATION_V2_RESOURCE_CLARIFICATION.md) revises the original 64 MiB evidence allowance to 1 GiB of host disk, not GPU VRAM.

The [DistilBERT agreement pilot](results/bert-agreement-dev-v0-20261028/README.md) observed donor-directed shifts after subject-state patches on four public development examples. All four used plural distractors, so the finding is confounded and a balanced evaluation is needed before making a mechanism claim.

The [DistilBERT GPU calibration](results/bert-gpu-calibration-v1-20261029/README.md) checks patch delivery and allocator cleanup on one previously exposed development pair. It is not a held-out evaluation or a GPU throughput benchmark.

The [DistilBERT GPU family profile](results/bert-gpu-family-profile-v1-20261029/README.md) records 140 forwards on one exposed family, with a 324 MiB peak PyTorch reservation and zero PyTorch allocator bytes allocated/reserved after cleanup. An earlier subject-state comparison produced a larger mean effect than the candidate site, so the result does not uniquely localize the computation. These allocator observations are not a worst-case VRAM bound.

The [balanced DistilBERT CPU evaluation](results/bert-agreement-cpu-evaluation-v1-20261029/README.md) tested the candidate on eight public families held out from local site selection. The candidate exceeded its matched-layer distractor control, but an earlier subject-state patch produced a larger effect in all eight families. The cases were public rather than blind, and the result does not establish a uniquely localized grammatical-number representation.

The [DistilBERT exposed-pair GPU observation](results/bert-gpu-exposed-pair-v1-20261028/README.md) reduced the `is − are` margin from +5.2040 to +0.0572 on one previously exposed public pair, without reversing that two-word preference. Its peak PyTorch reserved memory was 324 MiB; this allocator counter does not measure total or worst-case GPU memory.

The [DistilGPT2 agreement preflight](results/distilgpt2-agreement-preflight-v1-20261028/README.md) found the grammatical verb preferred on all 16 fixed public development prompts; it establishes a small behavior to investigate, not a held-out or causal result.

The independently audited [agreement discovery](results/distilgpt2-natural-agreement-v1-20261028/README.md) selected the layer 3 attention output at the final token. Across four public families, opposite-number donors caused a mean 3.095-logit greater shift away from the target's present-tense agreement margin than same-number donors. They reversed the margin in 10 of 32 donor–target pairs involving six distinct targets, versus none of 32 same-number pairs. These are development results, and a shift can reflect weakening rather than transfer of number information; held-out prediction remains to be tested.

An [independent held-out auditor](docs/INDEPENDENT_NATURAL_HELDOUT_AUDIT.md) is prepared to check saved controls and frozen-prediction errors after reveal; it does not establish that a held-out study has run.

The [DistilGPT2 copying screen](results/distilgpt2-copying-dev-v1-20261028/README.md) found higher target log probability after repetition than after each of three first-block shuffles in eight public development families. The shuffles also change local transitions; the observed attention pattern proposes a head for intervention but does not establish a cause.

In a [follow-up intervention](results/distilgpt2-copy-head-eval-v1-20261028/README.md) on eight new public families, zeroing the selected head at the final input position reduced the repeated-versus-shuffled log-probability contrast by 0.638 nats on average, compared with an increase of 0.130 for one same-layer control head. All eight repeated targets remained top-ranked after the selected-head intervention. These fixed-order exploratory tests do not establish a complete copying circuit.

A [broader head comparison](results/distilgpt2-copy-head-followup-v2-20261028/README.md) found a larger mean contrast reduction from another head in seven of eight families, failing the proposed specificity criterion for the originally selected head. The interventions have different activation magnitudes, and these public exploratory results do not establish a unique copying mechanism.

The [matched-dose copying factorial](results/distilgpt2-copy-head-factorial-v1-20261028/README.md) compared single-head and joint interventions on 32 public prompts. Both candidate heads affected the repeated-versus-shuffled contrast, with variation across families. The joint intervention had a larger total input dose than either single intervention, and downstream nonlinearities can contribute to the observed output interaction; the result does not establish direct cooperation between heads.

A [targeted parameter-edit experiment](results/targeted-edit-cpu-v1-20261028/README.md) changed the intended answer preference but failed its preset locality thresholds on public probes. The negative result and a same-host replay are preserved; the study does not establish general capability preservation.

The preserved v1 runner has a [cleanup compatibility limitation](docs/TARGETED_EDIT_V2_OWNER.md#historical-v1-cleanup-limitations) in Python 3.10 and when called while an unrelated exception is being handled. The recorded run used Python 3.11; this limitation does not reclassify its result.

## Run locally

Python 3.10+ and PyTorch 2.6+ are required. Use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your platform. The original CPU experiment downloads no models or datasets, and traces stay in the local process/files.

The optional `transformer` extra pins Transformers 4.57.3: install it with `python -m pip install -e ".[transformer]"` after selecting the appropriate PyTorch build. The [DistilGPT2 loader](docs/PRETRAINED_MODEL.md) is offline by default; downloading the pinned checkpoint requires an explicit opt-in and cache directory. Its CPU smoke test checks loading and execution, not causal explanations or performance. The original CPU experiment does not require this extra or a checkpoint download.

The [recorded CPU shared-service checks](docs/PRETRAINED_SERVICE_SMOKE.md#recorded-cpu-checks) compare full DistilGPT2 logits for two clients against serial references with one model deserialization. They check correctness for the specified inputs and serial execution, not causal explanations or GPU performance.

The [pretrained trace smoke](results/pretrained-trace-cpu-v1-20261028/README.md) records 18 events per channel on one fixed five-token input and two fresh-process replays on the same CPU host. It saves hashes and metadata rather than raw activations; it is a correctness check, not a causal study or benchmark.

The [pretrained intervention smoke](results/distilgpt2-intervention-dev-cpu-v1-20261028/README.md) checks a known injection and same-site activation restoration on one public input, with an independent same-host replay. It tests intervention mechanics, not a held-out behavioral explanation.

The [sampled pretrained batch check](results/distilgpt2-batch-cpu-v1-20261028/RESULTS.md) compares serial and two-row outputs on four fixed public CPU cases against a prespecified tolerance, with negative controls and a same-host replay. It does not establish general batch separability or GPU throughput.

The [grouped-broker CPU development replay](results/grouped-broker-cpu-dev-v3/README.md) exercises team routing and serial-versus-grouped logits on those same four cases. It is not independent held-out evidence or a GPU performance result.

The recorded original experiment used Python 3.12.14 and PyTorch 2.15.0+cpu on Linux. [requirements-tested.txt](requirements-tested.txt) records that run's installed versions; it is not a cross-platform dependency lock. Later experiments record their own environments. To install the original run's PyTorch version in an isolated environment, use `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

Install the test and transformer extras and use pytest to discover both
`unittest` cases and pytest-style tests. Some checks require optional local
model assets and will be skipped when those assets are unavailable. Set
`CUDA_VISIBLE_DEVICES=-1` before testing to keep the run CPU-only.

```bash
python -m pip install -e ".[test,transformer]"
python -m pytest -q tests
python -m looking_glass.experiment --output results/my-full-run
```

`unittest discover` can still run individual unittest-based files, but it
does not collect the module-level pytest tests and is not the full test suite.

For a short smoke check, use `--quick`. It is explicitly labeled as not the full protocol. Existing output directories are never overwritten. Full results contain raw timing samples, CSV summaries, numerical checks, source hashes and environment versions. Once evaluation starts, a failed run writes a failure record rather than a successful summary. Missing dependencies fail at import with an error on stderr, before a run directory is created.

Run the experiment as a standalone process: it seeds PyTorch's RNG and sets its intra-op thread count to one. Active CPU autocast is rejected rather than silently mislabeling mixed-precision data. Timing covers forward execution and the complete observation-session lifecycle, not task backward or optimizer latency.

## Use the observation API

```python
from looking_glass.model import TinyResidual
from looking_glass.probes import TraceSession
import torch

model = TinyResidual(width=16)
x = torch.randn(8, 16)
with TraceSession(model, mode="gradient", placement="dense", run_id="example",
                  max_tensor_bytes=65536) as trace:
    y = model(x, route=0)
for event in trace.events:
    print(event.metadata())
    activation_copy = event.value
```

`TapPoint` is an identity module that can be inserted at explicit model boundaries. Sparse placement records block outputs; dense placement records every available tap. `TraceSession` is single-use and single-threaded. Concurrent tracing, checkpoint recomputation, compiled execution and distributed models are outside this initial implementation. Configured event and cumulative captured-tensor-byte limits raise on overflow; capture errors are not silently ignored. The byte allowance covers logical snapshots, not temporary allocations or peak memory. Direct and gradient capture require finite floating tensors.

Full activations can contain sensitive information. This synthetic benchmark saves no input data, model weights or activation tensors in its result files. If you use the API on a real model, control who can access the in-memory traces and any files you write; the taps are not a privacy mechanism.

## Research notes

- [Mathematical identity and provenance limits](docs/MATHEMATICS.md)
- [Prespecified experiment](PROTOCOL.md)
- [Protocol clarifications and run history](docs/PROTOCOL_NOTES.md)
- [Independent review and corrections](REVIEW.md)
- [Audit of the preserved CPU records and reproducibility limits](docs/REPRODUCIBILITY_AUDIT.md)
- [Causal intervention literature and controls](docs/CAUSAL_LITERATURE.md)
- [Machine and GPU inventory](docs/GPU_INVENTORY_V1.md)
- [Supervised GPU memory pilot and recovery record](docs/GPU_ONESHOT_RESULTS_V2.md)
- [Exploratory direct-versus-gradient GPU timing](docs/GPU_TAP_LATENCY_RESULTS.md)
- [Site-blind injection selection and controls](docs/PRETRAINED_NUMBER_INJECTION_SELECTION.md)
- [Trace custody integration plan](docs/TRACE_CUSTODY_PLAN.md)
- [Shared-model throughput protocol](docs/THROUGHPUT_PROTOCOL_V1.md), its [CPU functionality record](results/throughput-cpu-smoke-v1-20261028/run_status.json), and the [quick GPU development comparison](results/throughput-gpu-quick-dev-v1-20261029/README.md)
- [Verification status](evidence/STATUS.md)
- [Design](docs/superpowers/specs/2026-10-28-runtime-probes-design.md)
- [Implementation plan](docs/superpowers/plans/2026-10-28-runtime-probes.md)

This work grew out of the [Trojan Regularizer Audit](https://github.com/EastonSell/turing-complete-malware-pytorch-backward-pass-audit), but addresses trusted runtime instrumentation. It is not a general backdoor detector or model-safety certification. Ordinary PyTorch hooks and the gradient identity are standard mechanisms; possible value lies in the measured placement, provenance and operating costs, not a novelty claim for those mechanisms.
