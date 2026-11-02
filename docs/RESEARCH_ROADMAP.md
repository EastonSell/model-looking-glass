# Research roadmap and execution efficiency

Living plan, updated October 30, 2026. This is a direction for future work, not a release claim or authorization to launch experiments. Frozen protocols, source hashes and historical outcomes are unchanged. See [project scope](PROJECT_STATUS.md), [recorded validation](MAINTENANCE_2026-10-29.md), and the [original results](../RESULTS.md) for what exists and what has actually been checked.

## Goal and measure of progress

Build a **causal debugger for neural networks**: observe relevant computations, generate competing explanations, predict interventions on new inputs, and measure the benefits and collateral effects of targeted changes. The long-term scope includes different architectures, multimodal models, generation, training and distributed execution; each extension needs its own evidence.

The decisive outcome is **accurate held-out intervention predictions and reproducible findings per unit of compute**, not tensor volume, agent count, GPU utilization or the plausibility of a visual explanation. An honest negative result or demonstrated inability to distinguish hypotheses is useful progress.

Keep three things separate:

- **Execution coordinates:** which instrumented operations ran, including layer, token, forward, branch and invocation identities where available.
- **Measurements:** values or summaries with declared capture coverage, numerical fidelity and completion status.
- **Causal hypotheses:** predictions about specified interventions and conditions, with alternatives and uncertainty.

A value is not a route label: different branches can produce identical vectors. A gradient is sensitivity to an objective, not a complete causal explanation. Ablations can create off-distribution states. Direct capture is the observation baseline; retain an auxiliary-gradient channel only if it earns its additional cost for a demonstrated purpose. The [denormal counterexample](GRADIENT_DENORMAL_V1.md) makes raw-bit controls important when bitwise fidelity matters.

## Current starting point

Use the existing [known-corruption and restoration workflow](GPT2_TRACE_WORKFLOW.md), [intervention plans](INTERVENTION_PLANS.md), [trace schema](TRACE_SCHEMA.md), [localization reference](LOCALIZATION.md) and [causal metrics](CAUSAL_METRICS.md), rather than rebuilding those components. The original localization API consumes original `TraceEvent` streams, not arbitrary transformer events; preserve coordinates through the appropriate model-specific comparison path.

`TraceSession` is local, single-use and single-threaded. The documented eager GPT-2, DistilBERT, T5 and ViT adapters have model-specific limits; they do not establish universal architecture or compiled-model support. Capture allowances count recorded payloads, not total process RAM or VRAM.

Broker, IPC, resident-worker and GPU scheduling components remain experimental infrastructure. The [grouped GPU development run](../results/grouped-gpu-resource-v3-20261029/README.md) reported fixed serial/two-row functional and resource observations, not shared-service throughput or a general memory-capacity bound. A complete default CPU test pass is not GPU performance evidence, and [experimental opcode probes](EXPERIMENTAL_OPCODE_TESTS.md) are not supported arbitrary-interruption guarantees.

## Current priorities

1. **Make the existing debugger workflow reproducible end to end.** Keep one usable path from a declared model/input and capture policy to controlled intervention comparisons, raw artifacts and a bounded conclusion. Report missing coordinates, truncation and unavailable values rather than presenting them as equality.
2. **Test prospective predictions, not another inventory of effects.** Select a behavior and competing mechanisms on discovery inputs; freeze candidate sites, interventions, metrics and predictions before confirmation. Use independent held-out units, matched controls, task-level/no-instrumentation baselines and simple random or magnitude-ranked alternatives. Preserve ineligible baselines, locality failures and negative findings.
3. **Measure observation value against cost.** Compare sparse, dense, random and adaptive placement at equal location and byte/time budgets. Report missed changes, localization interval width and coverage. A narrower first-observed interval is not proof of a unique cause.
4. **Profile the exact GPU workload where execution is authorized.** Compare uninstrumented and instrumented runs, including capture validation, copies and required output handling. Reuse existing runners where suitable; require a new scheduler or service feature to solve a measured blocker, not merely create another preparation stage.
5. **Expand only with acceptance evidence.** Add generation/cache coordinates, other architectures, compiled execution, training or multiple devices as distinct compatibility tracks. For targeted edits, require rollback/restoration checks and independently held-out collateral-behavior measurements, not only improvement on the target score.

These priorities do not cap the research ambition or number of contributors. Independent work can proceed in parallel; dependent experiments need compatible inputs and evidence. No central approval queue or fixed team organization is prescribed here.

## Experiment records and efficient reuse

Each run should record the hypothesis and alternatives; source/protocol versions; immutable model, tokenizer and data identities; input/case IDs; runtime, dtype, mode and RNG policy; intervention and capture policy; resource/work limits; completion criteria; and artifact destination. Distinguish pending, completed, failed, cancelled and incomplete runs. Keep raw observations and failures alongside summaries, so another investigator can reproduce the conclusion without unstated assumptions.

Reuse an artifact only when the full experimental identity matches, including adapters, generation settings and relevant backend configuration. A matching prompt string is insufficient. Never substitute a clean-run activation for an edited or differently conditioned forward. Same-forward controls must actually use that forward. Private prompts and activations stay out of public artifacts by default.

Before adding concurrency, consider cheaper work: deduplicate genuinely identical runs; narrow explicit token/layer windows; compare coarse-to-fine taps with equal-budget baselines; use bounded offload; or selectively recompute reproducible states. Compression and sketches need error contracts or measured task-specific fidelity and exact controls. Unrecorded positions are not evidence that no change occurred. Freeze adaptive selection rules before held-out evaluation.

## Memory and admission

Verify the actual device, available memory, driver/runtime, operating system and supported dtypes before execution. Profile the exact workload: weights, activations, generation caches, gradients/optimizer state where applicable, trace copies, temporary validation tensors, and allocator/runtime overhead. An old profile for a different input shape, capture mode or runtime is not a new workload's bound.

For a future admitted service, check both resident allocations plus conservative incremental work and reserve against the configured budget, **and** current device-wide free memory against additional allocations and reserve. These are estimates and a momentary observation, not an OOM guarantee. Measure device-wide free memory separately from process allocated/reserved memory; tensor-memory peaks are not the complete device footprint. Choose and record headroom from observed transients and other applications, not an unverified universal percentage.

Bound events, captured positions, tensor bytes, host RAM, pinned buffers, disk output, context length and elapsed work. `max_events` alone cannot bound bytes. A first estimate for one dense activation family is:

```text
batch × captured_tokens × hidden_width × element_bytes × saved_layers
```

Additional tensor families, gradients, clones and transfer buffers add to that estimate. Host offload consumes bandwidth and memory; RAM is not interchangeable with VRAM. Keep ownership until device work and required transfers complete.

On exhaustion or uncertain cleanup, retain failure evidence, mark the run incomplete and recover only owned resources. Any retry is a new labeled run with its changed configuration recorded. Do not silently drop taps, reduce precision or shorten inputs and call the original run successful. If process state cannot be trusted, require verified process recovery rather than continuing allocations. Preserve completed artifacts and unrelated applications.

## Timing and useful concurrency

GPU enqueue time is not end-to-end latency. Use synchronized wall-clock measurements and device events for explicitly defined spans; joins must cover measured work on every participating stream. Report queue wait, service time and total response time separately. If completion promises host-resident snapshots or written artifacts, include those operations. Keep raw repeated samples and runtime/configuration identifiers.

The original probe code converts a finite-value reduction to a Python boolean at value taps; on CUDA this host decision introduces synchronization. Measure validation, cloning, metadata, gradient readout and offload separately. An asynchronous alternative must retain a defined validation boundary and reject invalid traces, not omit checks to improve timings.

Keep two protocols distinct:

| Measurement | Required comparison |
|---|---|
| Isolated latency/memory | Reserve the measured workload, account for unrelated activity, warm up each cell, and randomize/interleave repeated baseline and instrumented runs. Prespecify how contaminated intervals are labeled or excluded; do not stop unrelated applications. |
| Shared-workload throughput | Fix job mix, arrival rates, concurrency and budgets; compare with the same serial workload. Report successful work per second, response-time distribution, queue delay, fairness, failures and memory. This is not uncontended per-job latency. |

Reuse a resident checkpoint only with a verified compatible model/configuration identity. Prefer batching validated against a serial reference before assuming streams or additional workers will help. Queued requests are not simultaneous GPU execution; separate streams do not guarantee hardware overlap. Increase concurrency only when measured valid throughput justifies its latency and memory cost.

Do not overlap independent sessions on one mutable model by installing competing global hooks. Shared execution would need explicit per-request ownership and sample/token/invocation coordinates, including request-local cache, input, gradient and capture state. Test exceptions, cancellation, repeated modules, recomputation and cross-request leakage. An agent count is not a GPU job count.

Parameter edits, optimizer steps, mode changes and global RNG mutation need isolated replicas or exclusive execution. Before reuse, restore/reload and verify weights, buffers, mode and relevant RNG/runtime state; discard an instance whose restoration is uncertain. Exclusivity does not undo an edit. Activation interventions also require proven request isolation, and auxiliary gradient-readout leaves must stay outside model optimizers.

## Acceptance and reasons to change direction

- Observational modes must meet declared output/gradient invariance and fidelity criteria, including failure cleanup and bounded capture. Match tolerance to the scientific claim; exact-bit checks and numerical equality answer different questions.
- A proposed mechanism must outperform stated alternatives on frozen held-out predictions. A targeted change must satisfy separate target and collateral-effect criteria; a successful patch alone is not an explanation.
- If a channel adds no useful evidence at greater cost, prefer the cheaper channel. If intervention predictions fail, revise the mechanism rather than expanding its infrastructure.
- If concurrency breaks correctness, isolate the incompatible workload. If it only worsens throughput, reduce concurrency according to measurements.
- If an effect disappears under independent replay or held-out testing, preserve the failed hypothesis. Automated experiment selection must be compared with simpler search under matched compute and review budgets, including false discoveries and wasted runs.

The next deliverable should answer a scientific or usability question with reproducible evidence. More scaffolding is justified only by a concrete missing capability on that path.
