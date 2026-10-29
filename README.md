# Model Looking Glass

**Mission: build a causal debugger for neural networks that can discover how a behavior is produced, test that explanation, and help change the behavior precisely.**

The long-term goal is to turn internal observations into explanations that predict interventions on unseen inputs, then use that understanding for targeted model improvements. Language, vision, multimodal systems, training and inference are all within the research direction. The current small experiment is the first instrument, not the ceiling.

- [Research roadmap: observation → tracing → tested explanations → targeted changes](docs/RESEARCH_ROADMAP.md)
- [Efficiency plan: parallel agent teams and a shared GPU experiment service](docs/EFFICIENCY_PLAN.md)

The roadmap is a research plan, not a claim that its milestones are complete. The original `TraceSession` remains single-threaded. Separate components provide bounded trace snapshots, trace metadata, experiment records, localization, causal metrics, and GPU scheduling primitives. A fixed-input, serial two-client GPU bootstrap has been exercised; a general shared GPU service has not yet been validated. No pretrained-model study, tested causal explanation, or model repair result is claimed by the initial experiment.

## Current building blocks

- The [trace schema](docs/TRACE_SCHEMA.md) records execution coordinates separately from tensor values and hypotheses. Unknown coordinates remain unknown.
- The [GPT-2 trace adapter](docs/TRANSFORMER_TRACE.md) captures selected layer, module, local token and invocation coordinates; generation and cache positions remain unknown unless established separately. The [ordered intervention plan](docs/INTERVENTION_PLANS.md) can capture, inject and patch at specified boundaries.
- The [DistilBERT masked-LM trace adapter](docs/DISTILBERT_TRACE.md) records bounded eager CPU snapshots at selected attention, FFN and block outputs. Its [intervention workflow](docs/DISTILBERT_INTERVENTION_WORKFLOW.md) tests a known artificial injection with restoration and an off-target comparison.
- The [T5 trace adapter](docs/FLAN_T5_TRACE.md) captures selected encoder and decoder outputs in one uncached forward with explicit masks. It does not trace generation or a separate cross-attention output. Its [donor-patch workflow](docs/FLAN_T5_INTERVENTION_WORKFLOW.md) checks specified state transfers and self/off-target controls on CPU.
- The [ViT trace adapter](docs/VIT_TRACE.md) captures bounded CLS and patch-token observations on CPU, with coordinates in the processed image grid. Its [donor-patch workflow](docs/VIT_DONOR_PATCH.md) checks specified CLS and patch-position transfers and controls on CPU. Neither infers original-image boxes or causal localization.
- The [fair queue](docs/FAIR_QUEUE.md) rotates among admitted teams; it does not by itself guarantee admission fairness or equal GPU time.
- The [memory budget](docs/MEMORY_BUDGET.md) and [resource profiles](docs/RESOURCE_PROFILE.md) track conservative leases, trace allowances and workload-specific measurements. Source-specific profiles were used to admit a fixed 1 × 2, zero-trace serial GPU bootstrap; they do not transfer to changed sources or workloads. Estimates and logical byte counts are not peak-memory guarantees or proof of device enforcement.
- The [resident worker](docs/RESIDENT_WORKER.md) and [broker](docs/BROKER.md) reuse one trusted model for queued forwards. Their current execution is serial; GPU-wide admission and concurrent device throughput require separate integration and validation.
- The [local CPU service](docs/SERVICE.md) exposes the pinned DistilGPT2 model to authenticated clients, with a two-token input limit and serial execution. A separate [microbatch primitive](docs/MICROBATCH.md) and [sampled batch validator](docs/BATCH_VALIDATION.md) are not yet integrated into that service.
- The [device supervisor](docs/SUPERVISOR.md) coordinates a cooperative owner process, a device lease, and a sticky quarantine marker. A [fixed two-client GPU bootstrap](results/serial-two-client-gpu-bootstrap-v1-20261029/README.md) exercised supervised serial execution; this is not validation of a general shared service.
- The [forward-only IPC v1](docs/IPC.md) lets authenticated cooperating client processes submit bounded forwards over loopback. It does not by itself enforce GPU budgets or provide trace delivery.
- The [experimental fair IPC bridge](docs/IPC_FAIR_BRIDGE.md) joins bounded forward-only admission and receipt custody in CPU tests. GPU service composition requires separate admission and supervision.
- The [trace delivery prototype](docs/TRACE_TRANSPORT.md) tests bounded, single-event custody on CPU. The original forward-only service does not deliver traces.
- The separate [CPU trace IPC v2](docs/TRACE_IPC_V2.md) delivers bounded multi-event traces to two trusted local client processes. Host budgets are independent per process and release is cooperative; it does not enable GPU tracing.
- A [CPU trace owner](docs/CPU_TRACE_OWNER.md) composes the [traced worker](docs/TRACED_WORKER.md), [producer envelope](docs/TRACE_PRODUCER.md) and [host custody ledger](docs/HOST_BUDGET.md) for private in-process jobs. It also underpins the separate CPU trace IPC v2; it is not integrated with the existing forward-only broker and does not impose a physical memory limit.
- The [experiment registry](docs/REGISTRY.md) records manifests, artifacts and terminal states. [Research cases](docs/RESEARCH_CASES.md), [localization](docs/LOCALIZATION.md), and [causal metrics](docs/CAUSAL_METRICS.md) support controlled evaluations; the public held-out cases are not a protected blind benchmark. [Transformer trace comparison](docs/TRANSFORMER_LOCALIZATION.md) requires independently verified pairing and execution context.
- The [behavioral edit evaluator](docs/BEHAVIORAL_EDIT_EVALUATION.md) reports paired target and locality observations, supplied distribution changes, and scoped rollback evidence. It does not establish semantic damage or prescribe success thresholds.

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

The [matched-dose follow-up](results/bert-matched-dose-v1-20261029/README.md) did not meet its preregistered early-site persistence rule: only four of eight family differences favored block 0 over block 2, versus six required. The result does not establish that the sites are equivalent. A separate [taste-context exploration](results/bert-taste-context-v1-20261029/RESULTS.md) found donor-directed ingredient-patch effects on four public sentence frames, although the clean `sweet`/`sour` preference was not reliable across frames.

In the [taste-context propagation follow-up](results/bert_taste_context_propagation_v1/RESULTS.md), the early ingredient intervention's contrast against two orthogonal controls was positive in all 24 recipient/template cells at each measured block-3 position. A direct block-3 `the` intervention shifted the taste score opposite its lexical donor in all 24 cells. These correlated public examples do not identify a unique route or a general taste representation.

On six public nonblind contexts from one template, [DistilBERT negation-cue transfers](results/bert-negation-cue-cpu-v1-20261029/README.md) met the directional `any`-versus-`some` logit rule (mean +0.2543, 6/6 positive). The reverse effect was not unique to the `just` donor: the absolute shift from the `also` donor was at least as large in 11/12 matched-recipient comparisons; this does not test polarity specificity. The compact repository does not contain the locally retained full journal.

The [cue-class follow-up](results/bert-cue-class-cpu-v1-20261029/README.md) passed its fixed specificity rule in all three frames: seven of nine new public evaluation contexts passed, and all nine were eligible. Each template also had a development context; the cues differ in lexical identity and frequency, so the result does not establish general negation understanding or unique localization. The full journal is locally retained but omitted from the repository.

The [held-out-frame follow-up](results/bert-cue-heldout-frames-cpu-v1-20261029/README.md) failed that specificity rule in all six eligible contexts; cue words and frames changed together. In the [cue-set × placement factorial](results/bert-cue-factorial-cpu-v1-20261029/README.md), all four cells failed the fixed rule (0/3 cases each), although 11 of 12 worst-target margins remained positive. The factorial used a lower common dose than the earlier study and changed the template, so it cannot determine the cause of the earlier difference or establish zero effect.

In a [matched-dose taste follow-up](results/bert-taste-matched-dose-v1-20261029/RESULTS.md), 29 of 30 directional comparisons across blocks 0–4 favored the ingredient patch over the position control. Only three of six frames met the baseline rule, all using the same template; the comparisons are dependent and do not establish a general taste mechanism.

The [crossed-template taste study](results/bert_taste_template_direction_v1/RESULTS.md) compared six public stems under two templates. Its mean ingredient-versus-`the` donor-signed contrasts were +0.085 for `after adding` and +0.068 for `with some`; the `the` control itself moved opposite the donor in all 24 comparisons. These dependent, nonblind comparisons do not establish a general taste mechanism.

The [DistilBERT exposed-pair GPU observation](results/bert-gpu-exposed-pair-v1-20261028/README.md) reduced the `is − are` margin from +5.2040 to +0.0572 on one previously exposed public pair, without reversing that two-word preference. Its peak PyTorch reserved memory was 324 MiB; this allocator counter does not measure total or worst-case GPU memory.

The [grouped DistilGPT2 GPU pilot](results/grouped-gpu-resource-v3-20261029/README.md) used one resident model for eight single-row and four two-row calls. The owner reported that the grouped samples matched their serial references; logits were not retained for independent recomputation. The largest recorded per-forward PyTorch allocated/reserved peaks were 353/388 MiB, and current allocated and reserved bytes were zero after cleanup. These sampled, shape-specific observations are not a throughput result or a general memory bound; activation tracing was disabled.

The [two-client GPU bootstrap](results/serial-two-client-gpu-bootstrap-v1-20261029/README.md) completed two fixed requests with overlapping outstanding client work but serial broker service. It did not retain serving-time GPU memory, independent full-logit correctness checks, or a matched throughput baseline. Its source-specific, zero-trace results do not establish sustained fairness, batching performance, or a general memory bound.

The [DistilGPT2 agreement preflight](results/distilgpt2-agreement-preflight-v1-20261028/README.md) found the grammatical verb preferred on all 16 fixed public development prompts; it establishes a small behavior to investigate, not a held-out or causal result.

The independently audited [agreement discovery](results/distilgpt2-natural-agreement-v1-20261028/README.md) selected the layer 3 attention output at the final token. Across four public families, opposite-number donors caused a mean 3.095-logit greater shift away from the target's present-tense agreement margin than same-number donors. They reversed the margin in 10 of 32 donor–target pairs involving six distinct targets, versus none of 32 same-number pairs. These are development results, and a shift can reflect weakening rather than transfer of number information; held-out prediction remains to be tested.

On fixed public nonblind cases, both exploratory [agreement-direction rules](results/distilgpt2-agreement-direction-study-v1/RESULTS.md) passed (8/8 families and 16/16 pairs). Subject-free but number-marked context fragments also showed a substantial response, so the result does not establish an abstract number feature or unique circuit.

The [subject-form sensitivity study](results/distilgpt2-agreement-subject-dependence-v1/RESULTS.md) found a larger signed response for the singular member in 31/32 public matched pairs, while subjectless fragments also responded. The cases are dependent and nonblind; morphology, lexical identity and context-dependent gain remain possible explanations.

The [two-attractor patch follow-up](results/public-two-attractor-patch-v1-20261029/README.md) found a near-zero pooled clean interaction that concealed opposite subject-specific means. Its donor displacements were unequal and some donors changed local lexical identity, so these descriptive results do not establish independent attractor channels or additivity.

The [natural-agreement held-out attempt](results/distilgpt2-natural-agreement-v1-20261028/heldout-attempt-status-v1.md) stopped without an evaluable result. Its one-use claim and sealed split remain preserved; the recorded RuntimeError and a no-model startup reproduction support, but do not prove, a missing dependency-path diagnosis. An [independent held-out auditor](docs/INDEPENDENT_NATURAL_HELDOUT_AUDIT.md) is prepared for saved controls and frozen-prediction errors after a valid reveal.

The [DistilGPT2 copying screen](results/distilgpt2-copying-dev-v1-20261028/README.md) found higher target log probability after repetition than after each of three first-block shuffles in eight public development families. The shuffles also change local transitions; the observed attention pattern proposes a head for intervention but does not establish a cause.

In a [follow-up intervention](results/distilgpt2-copy-head-eval-v1-20261028/README.md) on eight new public families, zeroing the selected head at the final input position reduced the repeated-versus-shuffled log-probability contrast by 0.638 nats on average, compared with an increase of 0.130 for one same-layer control head. All eight repeated targets remained top-ranked after the selected-head intervention. These fixed-order exploratory tests do not establish a complete copying circuit.

A [broader head comparison](results/distilgpt2-copy-head-followup-v2-20261028/README.md) found a larger mean contrast reduction from another head in seven of eight families, failing the proposed specificity criterion for the originally selected head. The interventions have different activation magnitudes, and these public exploratory results do not establish a unique copying mechanism.

A [post-hoc full-vocabulary analysis](results/distilgpt2-copy-head-followup-v2-20261028/posthoc-distribution/README.md) of the saved repeated-input logits found mean total variation of 0.18876 after zeroing head 2 versus 0.09230 for head 10; head 2 was larger in seven of eight families. The metrics were selected after the original results, and the interventions have unequal norms, so this is a descriptive comparison rather than isolated off-target change.

The [matched-dose copying factorial](results/distilgpt2-copy-head-factorial-v1-20261028/README.md) compared single-head and joint interventions on 32 public prompts. Both candidate heads affected the repeated-versus-shuffled contrast, with variation across families. The joint intervention had a larger total input dose than either single intervention, and downstream nonlinearities can contribute to the observed output interaction; the result does not establish direct cooperation between heads.

The [copying dose follow-up](results/distilgpt2-copy-head-pairs-dose-v1-20261028/README.md) found a mean high-dose contrast change of −0.138 nats for the H2+H10 pair on eight public families, with seven negative family effects and one positive. Target top-one status remained unchanged in all 480 intervention comparisons. Different total doses at matched per-head scale and output nonlinearities limit the interaction's interpretation; equal-total-dose singles and additive-logit comparators are also reported.

The [per-prompt matched-dose copying follow-up](results/distilgpt2-copy-head-prompt-dose-v1-20261029/README.md) found the prespecified head-2 versus head-10 advantage in four of eight families, below the 75% criterion. The fixed head-7 control and full-vocabulary effects are reported; matching removed pre-projection norms did not equalize downstream effects.

In a [recipient-name screen](results/distilgpt2-recipient-screen-v1-20261028/README.md), only three of 32 matched prompt pairs met the baseline rule, so the planned patches were not run. A separate [year-comparison screen](results/distilgpt2-greater-than-screen-v1-20261028/README.md) found a wording-dependent shift but a later-year preference even in all reverse and neutral prompts; it does not establish a general temporal comparator.

In the [year-transfer intervention](results/distilgpt2-year-transfer-v1-20261029/README.md), donorward shifts occurred under forward, neutral and reverse wording. The comparator baseline failed in all eight families, and only one met the separate forward-specific intervention criterion, below the six required. The result does not establish a numerical comparator or a unique circuit.

In the [explicit-year follow-up](results/distilgpt2-explicit-year-direction-v1-20261029/README.md), the strict comparator baseline failed in all eight pair-and-noun observations. A separately specified generic donorward-shift pattern passed in all eight under `after`, `before`, and `near` wording, relative to two matched-dose controls; the rotated control was itself active. The comparator intervention decision was ineligible, and the result does not identify a numerical-comparison circuit.

The [FLAN-T5 encoder study](results/flan-t5-equal-dose-cpu-v1-20261029/README.md) compared word-position patches with matched-dose instruction and punctuation controls. The preset rule was negative for both evaluable word pairs; the third pair failed the baseline and was not patched. Teacher-forced sequence scores include EOS and are sensitive to target length and prior frequency, so they are not generated translations or a unique translation mechanism.

The separate [FLAN-T5 dose-response attempt](results/flan-t5-dose-response-cpu-v1-20261029/README.md) was inconclusive: all six cases failed the baseline, and no patches ran. The differing target lengths are a possible contributor, not an established explanation.

In a [length-matched FLAN-T5 follow-up](results/flan-t5-equal-length-response-cpu-v1-20261029/README.md), four public word pairs met the fixed three-dose directional-response rule in both templates; two pairs were baseline-unsupported and were not patched. The largest matched-dose word-position patch reversed the original two-target preference in only one of 16 evaluable cell-direction comparisons. This is a small teacher-forced score study, not evidence of generated translation, generalization or a unique mechanism; equal token lengths do not remove lexical-prior differences.

The locally held-out [FLAN-T5 follow-up](results/flan-t5-heldout-response-cpu-v1-20261029/README.md) had an inconclusive primary result: two word pairs met its rule and four were unevaluable. A separately prespecified descriptive rule was positive on four pairs, negative on one and unevaluable on one. These teacher-forced results do not establish generalization or identify a unique computation; the preserved record includes an initial audit failure and its later correction.

The [pretrained ViT intervention study](results/vision-coco-vit-v1-20261029/README.md) found that none of four evaluable held-out image pairs met the preset rule; four other pairs were baseline-ineligible. The patches had nonzero effects and were large relative to recipient token-state norms. Same-species controls were available for only four of 16 receiver sides, and outside-box controls can include other objects, so the result does not establish the absence of visual effects or a uniquely localized object representation.

The [same-site small-dose ViT follow-up](results/vision-coco-same-site-v3-20261029/README.md) was inconclusive: two of eight pairs met the baseline gate and neither passed the composite. The reused cat-annotated donor was classified as a dog by the model, so these dependent, annotation-matched comparisons do not establish spatial localization.

The [recognized-donor ViT follow-up](results/vision-coco-recognized-v4-20261029/README.md) was negative under its preset composite: all four selected pairs were eligible, and none passed both directional side rules. The screen was conditioned on model recognition, and two cat donors were reused, so the comparisons are dependent and do not establish spatial localization.

In the [CLS versus spatial follow-up](results/vision-coco-cls-v5-20261029/README.md), both screened pairs were eligible but neither passed the fixed two-sided composite. Active controls sometimes had larger effects, and one spatial intervention changed top-1; the result does not imply no effect or equivalence. The pairs reused both donors and were conditioned on model recognition.

In the [spatial ViT dose-response follow-up](results/vision-coco-spatial-dose-v6-20261029/README.md), all four model-screened pairs were evaluable, but none met the fixed control-adjusted low/high-dose composite. Some opposite-donor effects grew with dose while matched same-species or random controls were larger; a shared cat donor makes the observations dependent. The result does not establish absence of spatial information or class-specific localization.

A [targeted parameter-edit experiment](results/targeted-edit-cpu-v1-20261028/README.md) changed the intended answer preference but failed its preset locality thresholds on public probes. The negative result and a same-host replay are preserved; the study does not establish general capability preservation.

The [projected-edit follow-up](results/targeted-edit-projected-cpu-v1-20261028/README.md) was ineligible because only two of four targets had the required baseline preference; neither reversed. The projected and unprojected edits had different target effects, and the nine locality probes cannot establish broad capability preservation.

In the [screened-edit follow-up](results/targeted-edit-screened-matched-v1-20261029/README.md), both preset projected-edit comparisons were negative on eight baseline-selected public targets and twelve locality prompts: one target margin reversed, and one locality top-token ID changed from ` euro` to ` Euro`. These fixed probes do not establish broad capability preservation.

The [full-vocabulary targeted-edit screen](results/targeted-edit-full-top-cpu-v1-20261029/screen/README.md) was ineligible: none of four facts met the fixed baseline selection rule across its public prompts. No edit or locality prompt was run, so this is a feasibility outcome rather than a negative edit result.

The [answer-format feasibility study](results/answer-format-feasibility-cpu-v1-20261029/README.md) also selected no edit target: neither development nor held-out family met the fixed full-vocabulary baseline. The held-out examples instead showed a strong response to the two supplied demonstrations; they reused the development facts and do not identify a copying mechanism.

The preserved v1 runner has a [cleanup compatibility limitation](docs/TARGETED_EDIT_V2_OWNER.md#historical-v1-cleanup-limitations) in Python 3.10 and when called while an unrelated exception is being handled. The recorded run used Python 3.11; this limitation does not reclassify its result.

## Run locally

Python 3.10+ and PyTorch 2.6+ are required. Use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your platform. The original CPU experiment downloads no models or datasets, and traces stay in the local process/files.

The optional `transformer` extra pins Transformers 4.57.3: install it with `python -m pip install -e ".[transformer]"` after selecting the appropriate PyTorch build. The [DistilGPT2 loader](docs/PRETRAINED_MODEL.md) is offline by default; downloading the pinned checkpoint requires an explicit opt-in and cache directory. Its CPU smoke test checks loading and execution, not causal explanations or performance. The original CPU experiment does not require this extra or a checkpoint download.

The [recorded CPU shared-service checks](docs/PRETRAINED_SERVICE_SMOKE.md#recorded-cpu-checks) compare full DistilGPT2 logits for two clients against serial references with one model deserialization. They check correctness for the specified inputs and serial execution, not causal explanations or GPU performance.

The [pretrained trace smoke](results/pretrained-trace-cpu-v1-20261028/README.md) records 18 events per channel on one fixed five-token input and two fresh-process replays on the same CPU host. It saves hashes and metadata rather than raw activations; it is a correctness check, not a causal study or benchmark.

The [pretrained intervention smoke](results/distilgpt2-intervention-dev-cpu-v1-20261028/README.md) checks a known injection and same-site activation restoration on one public input, with an independent same-host replay. It tests intervention mechanics, not a held-out behavioral explanation.

The [pretrained trace and intervention example](docs/PRETRAINED_TRACE_INTERVENTION_EXAMPLE.md) shows how to use the APIs directly on a short CPU input, with a bounded trace, self-patch control and saved explanatory output. Its known artificial edit demonstrates the tools; it is not an explanation of the model's ordinary behavior.

The [clean-versus-corrupted trace workflow](docs/GPT2_TRACE_WORKFLOW.md) compares selected observations after a known embedding edit and tests same-site restoration against controls. The example is a mechanics check, not a causal discovery.

The [sampled pretrained batch check](results/distilgpt2-batch-cpu-v1-20261028/RESULTS.md) compares serial and two-row outputs on four fixed public CPU cases against a prespecified tolerance, with negative controls and a same-host replay. It does not establish general batch separability or GPU throughput.

The [grouped-broker CPU development replay](results/grouped-broker-cpu-dev-v3/README.md) exercises team routing and serial-versus-grouped logits on those same four cases. It is not independent held-out evidence or a GPU performance result.

The recorded original experiment used Python 3.12.14 and PyTorch 2.15.0+cpu on Linux. [requirements-tested.txt](requirements-tested.txt) records that run's installed versions; it is not a cross-platform dependency lock. Later experiments record their own environments. To install the original run's PyTorch version in an isolated environment, use `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

Install the test and transformer extras and use pytest to discover both
`unittest` cases and pytest-style tests. The test extra pins Pillow 11.3.0
for the vision selectors' fixed polygon rasterization. Some checks require
optional local model assets and will be skipped when those assets are unavailable. Set
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




