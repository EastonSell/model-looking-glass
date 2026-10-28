# Parallel teams and GPU efficiency

Status: proposed design, October 28, 2026. No shared GPU service or concurrent tracing is implemented by this document. Today's `TraceSession` is single-use and single-threaded; do not attach multiple sessions to one live model and assume they are isolated. The published experiment remains CPU-only.

## Objective

Let many research teams make progress together and use the GPU concurrently when that produces more valid work. Optimize for **independently validated evidence per wall-clock hour and GPU-hour**, with correctness, latency and memory budgets. Raw utilization, agent count and number of generated experiments are supporting metrics, not the goal.

The initial hardware target is an RTX 2070 Super with nominal 8 GB VRAM. Verify the actual GPU, free memory, driver, operating system, PyTorch/CUDA build and supported dtypes before execution. No fixed model size or concurrency count is guaranteed to fit. Larger hardware can use the same job contracts with different measured budgets.

## Three kinds of parallelism

1. **Research parallelism:** dozens of agents can inspect literature, build adapters, formulate hypotheses, write tests and review results in parallel. Agent-service capacity is separate from the local GPU's experiment capacity; 50 agents do not imply 50 model copies or GPU jobs.
2. **Workload concurrency:** compatible requests from multiple teams can share a resident model, be batched, overlap device work where beneficial, or use separate admitted workers.
3. **Controlled measurement:** dedicated performance experiments temporarily reserve the GPU, while the other teams continue CPU analysis and preparation. Shared-service throughput experiments intentionally use contention and report that workload separately.

No team gets permanent ownership of the GPU. Each submits bounded jobs to a shared scheduler with fair allocation, queue aging and small-job opportunities. Users can pause or cancel the research workload without corrupting completed artifacts.

## Proposed architecture

- **Coordinator:** owns task dependencies, scientific protocols and review assignments.
- **Experiment registry:** deduplicates requests and stores immutable manifests, pending/completed/failed/cancelled states, lineage and hashes.
- **Local GPU broker:** owns CUDA contexts, model lifecycles and admission. Agents submit validated experiment specifications, not arbitrary mutations of a shared model. Start as a local process/IPC boundary, not a new public endpoint.
- **Resident-model pool:** reuses compatible immutable checkpoints. Model cache identity includes weights, adapter revision, architecture, dtype, device and relevant runtime configuration.
- **CPU workers:** prepare inputs, analyze completed traces, create visualizations and check results without holding GPU memory unnecessarily.
- **Artifact writer:** writes bounded outputs atomically, labels incomplete runs and enforces retention/privacy settings.

This is a design choice, not an existing server. Setup, installations and permissions still follow the user's authorization. The plan does not assume an MPS service, hardware partitions or a supported operating system configuration.

## How multiple teams share a model safely

Prefer one resident immutable model for compatible read-only forward jobs. Group requests by model/configuration, batch shape compatibility, recording policy and required numerical contract. Batched execution may differ numerically from separate execution, so verify it against the serial reference and preserve request boundaries.

Before admitting same-model overlap, replace per-session hook installation with a concurrency-aware dispatcher. Every event needs job, request, token/sample, generation-step, node and invocation identifiers. Each emitted request-level event or tensor slice must have exactly one owning capture context; a callback for a combined batch may dispatch several separately owned slices. Test repeated modules, cancellation, exceptions, checkpoint recomputation, and no cross-job event leakage. Do not let hooks from one team capture another team's outputs.

Mutable jobs need different handling:

- Parameter edits, optimizer steps, model-mode changes or global RNG changes cannot race with shared jobs. Use isolated replicas if memory permits, otherwise a temporary exclusive lease. Before returning that model to the immutable pool, restore or reload and verify its weights, buffers, mode and relevant runtime/RNG state against the pool identity; evict the instance if restoration cannot be established. Exclusivity prevents races but does not undo edits.
- Prefer functional activation interventions with request-local masks; prove isolation before combining them in a batch. Do not install global modifying hooks per team.
- Keep generation caches, inputs, gradients and optimizer state request-local. Sharing weights does not make the rest shareable.
- Keep the separate gradient-readout leaf outside every model optimizer, as in the initial API. Shared autograd execution and recording still require their own tests.

Start with controlled batched jobs and a serial reference. Add streams or additional workers only after measuring correctness and throughput. Multiple queued requests are not proof of simultaneous GPU execution; a stream is not guaranteed hardware overlap. Measure both admission concurrency and completed work.

## Admission and memory budgets

Model weights are only one part of VRAM. Profile each workload class for weights, activations, generation caches, task gradients/optimizer state if used, trace copies, temporary telemetry tensors and allocator/runtime overhead.

Use two checks at every admission:

1. Existing resident allocations plus conservative incremental job estimates plus a configured reserve must fit the usable device budget.
2. Current device-wide free memory must cover the new allocations and reserve, accounting for other applications.

These are estimates and a snapshot, not a guarantee against out-of-memory failures. The broker must also enforce runtime limits and react to changed device pressure. Track device-wide free memory separately from process allocated/reserved memory; PyTorch's tensor-memory peak is not the entire process or device footprint. [Memory APIs](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.mem_get_info.html), [allocated-memory peak](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html).

**Conservative starting policy to calibrate, not a measured optimum:** leave the larger of 1 GiB or 15% of reported total VRAM outside the research allocation budget, then increase headroom if desktop use or measured transients need it. Profile one small, short-context, batch-one job, then try two admitted compatible jobs or a combined batch. Increase concurrency only when the additional valid throughput is worth its latency and memory cost. A large training job may still need the whole research budget.

Set explicit limits for events, captured tokens, GPU trace bytes, host RAM, pinned buffers, disk bytes, context length and runtime. `max_events` alone does not bound bytes. For a single dense activation tensor family, `batch × tokens × hidden_width × element_bytes × saved_layers` is a useful first estimate; multiple tensor families and temporary/gradient buffers add to it. Avoid treating CPU offload as free or system RAM as equivalent to VRAM.

On memory exhaustion: mark the run incomplete, release its owned tensors, record the configuration and error, and retry only as a new explicitly labeled run with an allowed smaller configuration. Do not quietly drop taps, change precision, shorten prompts or call a failed run successful. Recover the worker in a fresh process if state cannot be trusted; preserve completed artifacts and unrelated user applications.

## Reduce work before adding concurrency

| Lever | Proposed use | What must be checked |
| --- | --- | --- |
| Model reuse | Keep one compatible checkpoint loaded across teams | No shared mutable state or stale checkpoint identity |
| Deduplicated runs | Reuse exact matching deterministic artifacts | Full model/data/code/protocol/config identity, not just a prompt string |
| Coarse-to-fine taps | Locate a region with metadata/cheap summaries, then capture nearby vectors | Missed detections and equal-budget baselines |
| Token/layer windows | Record only specified positions and intervals | Coverage is explicit; omissions are not negative evidence |
| Compression/sketches | Reduce telemetry volume for candidate screening | Error bounds or measured task-specific fidelity; keep exact controls |
| Batched requests | Amortize model execution for compatible teams | Per-request outputs, metadata and intervention isolation |
| Bounded offload | Move finished snapshots to host storage | Transfer completion, buffer lifetime, host/pinned-memory limits |
| Recompute selectively | Save less and rerun a narrow region | Reproducible state and measured time–memory tradeoff |
| Adaptive experiments | Spend runs where competing hypotheses disagree | Selection rule frozen before held-out evaluation |

Cache keys should include checkpoint and tokenizer revisions, input/dataset hashes, model mode, dtype, seeds/RNG policy, generation settings, adapters, interventions, tap schema, capture policy, code/protocol versions and relevant runtime/backend settings. Never reuse a clean-run activation as if it came from an edited or differently conditioned run. Store permitted data only; private prompts and activations stay out of public evidence by default.

## CUDA profiling and synchronization

GPU execution is asynchronous, so CPU enqueue time is not end-to-end latency. Use correctly synchronized wall-clock intervals and CUDA events for defined device spans; for multiple streams, events need dependency joins covering all measured work. Report queue wait, service time and total response time separately. Stream dependencies and buffer lifetimes must be explicit. [PyTorch CUDA execution semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution).

The current code converts a finite-value reduction to a Python boolean at each value tap. On CUDA that host decision is a synchronization point. Profile validation, cloning, gradient readout, metadata and offload separately. An asynchronous validation design may aggregate device flags and check them at a defined boundary, but must not release invalid traces as successful or omit failure detection merely to improve timings.

Performance experiments have two protocols:

- **Isolated latency/memory:** drain this broker's queue, reserve the device workload, record other GPU activity, warm up every cell, randomize/interleave baseline and instrumented cells, and collect repeated raw samples. If unrelated activity cannot be controlled, label or reject the comparison according to a prespecified rule rather than stopping unrelated processes.
- **Shared-service throughput:** fix the mix of teams, job classes, arrival rates, concurrency and budgets; report successful jobs/tokens per second, p50/p95 response time, queue delay, fairness, failure rate and peak memory. Compare against the same workload served serially. Never present these numbers as uncontended per-job latency.

Measure end-to-end work consistently: if a mode promises offloaded/validated traces, its completion time includes that work. Separate forward-only, forward-plus-capture, task backward and full training-step benchmarks. Preserve all raw samples and the environmental conditions. Seeds improve reproducibility but do not promise identical results across releases or devices; record deterministic settings and their cost. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Team operating contracts

Each job records: hypothesis, owner/team, source commit, immutable model/data identifiers, protocol version, input or dataset reference, seed and runtime settings, expected resources, maximum work, capture policy, intervention, completion criteria and artifact destination.

Use a bounded queue with priorities, fair per-team credits, timeouts, cancellation and backpressure. Coalesce identical jobs only when their data access and intended outputs permit it. A job obtains a lease before allocations and releases it only after device work and artifact handling are accounted for. No team bypasses the broker to launch an untracked CUDA process.

Run discovery, confirmation and independent reproduction as separate stages. Freeze candidate hypotheses before confirmation. Reviewers can access the run specification and artifacts without changing the original result. More concurrent agents must not become more untracked multiple comparisons.

## Implementation order and release gates

1. **Inventory and serial baseline:** verify hardware/runtime, profile one model and record memory/latency envelopes. No GPU evidence exists yet.
2. **Manifest and broker skeleton:** bounded queue, request schema, leases and cancellation; test with simulated resource budgets and CPU workers first.
3. **Single GPU worker:** checkpoint reuse, measured memory admission and deterministic cleanup. Fail closed on unsupported jobs.
4. **Compatible batching:** request-local traces, exact or prespecified tolerance checks against serial runs, no cross-request leakage.
5. **Concurrent workloads:** test streams or multiple workers when useful, with bounded memory and fair scheduling. Compare throughput and tail latency with batching and serial execution.
6. **Scientific workflow:** add adaptive placement and intervention jobs, with discovery/held-out separation and independent replay.
7. **Scale out:** additional GPUs or machines use the same manifests and evidence requirements. Do not assume multi-device results from a single-card test.

A release must pass cancellation/exception/OOM cleanup tests, output and gradient invariance for observational modes, request isolation, accurate accounting and reproducible replay. Hardware-specific speedups are accepted only after measurements on that hardware. The research teams can prepare these components in parallel; dependent GPU experiments wait for the corresponding correctness gate.
