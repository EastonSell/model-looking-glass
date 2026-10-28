# Research roadmap

Status: proposed research program, October 28, 2026. This expands the project's goals; it does not change the frozen [initial protocol](../PROTOCOL.md), measured [results](../RESULTS.md), or released API.

## The goal

Build a causal debugger for neural networks: a system that helps researchers discover how a behavior is produced, test competing explanations, and make targeted changes with measured consequences.

An investigator should eventually be able to give it a behavior of interest and receive:

1. A reproducible trace of relevant internal computations and execution metadata.
2. Candidate mechanisms with explicit competing explanations and uncertainty.
3. Predictions about what specified interventions will do on new inputs.
4. Experiments that test those predictions against appropriate controls.
5. A proposed change, its observed effect, and the capabilities it preserves or damages.

The decisive outcome is **accurate intervention predictions on held-out cases**, not the number of tensors collected or the plausibility of a visual story. A useful system must also recognize when the available observations cannot distinguish explanations.

This direction is open to larger models, different architectures, multimodal representations, agents built around models, and learning dynamics. Hardware and finite experiments determine the next step, not the ultimate scope. Each expansion needs its own evidence.

## Scientific questions

- **Observability:** What must be recorded to distinguish executions, recover relevant internal states, and localize a change?
- **Information flow:** Which transformations preserve, mix, suppress or expose information about a task? How much of that conclusion survives a change of representation or measurement method?
- **Mechanism discovery:** Can observed patterns generate useful intervention hypotheses more efficiently than simple baselines?
- **Causal testing:** Can a proposed mechanism predict the sign, magnitude and context dependence of a controlled intervention's effect?
- **Targeted change:** Can a localized activation intervention or parameter change produce a desired behavior while limiting unwanted changes on a separately held-out capability suite?
- **Efficiency:** Which observation and experiment policies yield more independently validated findings per unit of GPU time, memory and human review?

## A chain of evidence

Keep three objects separate throughout the system:

- **Execution graph:** which instrumented operations ran, with layer, token, invocation and branch identifiers.
- **Measurements:** activation values, gradients for a specified objective, summaries, or reconstructed/compressed observations with an error contract.
- **Causal hypotheses:** predictions about interventions under stated experimental conditions.

A value is not a route label. Identical outputs from different branches defeat value-only provenance. A gradient is a local sensitivity to an objective, not a complete causal account. Ablating a component can also create an off-distribution state. These are reasons to design controls, not reasons to stop at observation.

Candidate gate designs, direct hooks, isolated gradient readout, learned probes and compressed telemetry compete as instruments. No method gets privileged because it inspired the project. Keep a gradient channel only where it earns its cost in a defined setting. The current experiment finds no information advantage over direct capture.

## Milestones and acceptance evidence

Every milestone produces a versioned protocol, code, raw results, failure records and an independent reproduction. The following are research deliverables, not a promised schedule.

### M1. Trustworthy observation on pretrained models

Add adapters for attention, MLP and residual boundaries; token positions, generation steps, cache state and repeated invocations. Define what each recorded value represents and where intervention hooks would act.

Acceptance: instrumented and baseline runs agree under prespecified criteria for outputs, loss, relevant gradients and optimizer updates where training is tested. Require exact equality where the execution path permits it; specify and justify numerical tolerances before evaluation otherwise. Include exception cleanup, overflow, repeated modules, nonfinite values, trace truncation and replay tests. Start on a small model for fast falsification, then test additional families rather than generalizing from one architecture.

### M2. Recover execution and find changes efficiently

Compare ID-only logs, sparse fixed taps, dense taps, random budget-matched placements, and adaptive placement. Include branching, reused modules, residual merges and generation.

Acceptance: report route-label accuracy against independent execution ground truth, coverage and missing events, localization interval width, and a cost–accuracy frontier. Run both equal-location and equal-byte/time-budget comparisons. A small localization interval is not by itself an identified mechanism.

### M3. Discover candidate mechanisms

Combine traces with activation patching, structured ablation, feature-level probes and explicit alternative hypotheses. Use synthetic tasks with known internal mechanisms to validate discovery tooling, then pretrained-model behaviors with no claimed ground-truth mechanism.

Acceptance: beat simple alternatives on prespecified held-out intervention prediction metrics at a matched experiment budget. Compare against random locations, magnitude-ranked and gradient-ranked candidates, and a no-instrumentation task-level baseline where applicable. Include negative controls, null interventions and identical-vector/different-route cases. Candidate selection and threshold tuning use discovery data only.

### M4. Test explanations under intervention

Before seeing held-out outcomes, save each candidate's predicted effect, uncertainty, intervention location and values, task metric and evaluation set. Test multiple intervention strengths and contexts, including rescue/restoration controls and plausible replacement activations where available.

Acceptance: quantify prediction error, sign accuracy, calibration and failure rates against baselines; report uncertainty at the independent prompt/task or seed unit, not by pretending correlated tokens are independent samples. Replicate across prompts and model seeds/checkpoints where available. Label necessity and sufficiency only relative to the tested intervention and task, and report interactions or redundant pathways.

### M5. Targeted behavioral changes

Compare localized runtime interventions and small parameter edits with broader alternatives. Optimize on development examples, then freeze the change before testing it on new target examples and unrelated capabilities.

Acceptance: show the tradeoff between target improvement and collateral changes, including regression tests, distribution shifts, reversibility and reproducibility. A successful edit on one behavior is evidence for that case, not a universal repair method.

### M6. Automated research and broader systems

Let coordinated agents propose experiments, choose probes, challenge mechanisms and reproduce findings using a shared artifact registry. Extend to multimodal and distributed models, compiled execution and training trajectories in separate compatibility tracks.

Acceptance: at fixed compute and review budgets, compare agent-guided research with scripted search and simpler heuristics. Measure independently reproduced findings, false discoveries, experiment waste and elapsed time. Audit selection bias: retain failed runs and hypotheses that were rejected. Require human-reviewable evidence rather than an agent consensus vote.

## Parallel research teams

Organize by owned interfaces and falsifiable questions, not one uncontrolled copy of the whole project per agent. An environment that supports 50 or more agents can allocate them across these streams, with subteams for implementation, experiments and replication. The number below is a division of responsibility, not a cap on people or agents.

| Stream | Owns | First useful deliverable |
| --- | --- | --- |
| Model adapters | Tensor meaning, token/cache coordinates, architecture compatibility | One pretrained-model trace adapter and replay tests |
| Runtime and GPU | Broker, memory accounting, concurrency and timing | Profiled job admission with a serial reference |
| Probe strategy | Placement, summaries, compression, capture budgets | Fixed/random/adaptive placement comparison |
| Mechanism discovery | Hypothesis generation and interventions | Controlled discovery task with withheld interventions |
| Behavioral change | Targeted edits and collateral metrics | Frozen edit evaluation and rollback artifact |
| Falsification and reproduction | Independent checks and negative controls | Blind reproduction with discrepancy report |

One integration owner maintains interface contracts and the experiment registry. Teams use separate branches/workspaces, publish small reviewable changes, and do not rewrite a shared working tree. An independent team must be able to reproduce an artifact without relying on the author's unstated assumptions.

## The next implementation wave

1. Specify the trace/event schema and immutable run manifest before separate adapters diverge.
2. Build the CUDA timing and memory harness and a pretrained-model adapter in parallel.
3. Build the GPU broker's scheduler and concurrency-isolation tests alongside an independent reviewer.
4. Establish a serial pretrained-model correctness baseline before admitting shared-model concurrent workloads.
5. Run a budget-matched placement experiment and one held-out intervention-prediction task.
6. Publish the result and failure cases, then choose the next experiment using evidence rather than adding every proposed feature.

The [efficiency plan](EFFICIENCY_PLAN.md) defines the proposed operating model. This document does not launch those teams, configure the user's computer, download models, or start new experiments.

## What changes direction

- If a reporting channel yields the same evidence at greater cost, prefer the cheaper channel unless another tested constraint changes the comparison.
- If traces predict correlations but interventions fail, revise or reject the mechanism claim.
- If compression destroys the signal needed for the task, increase fidelity or narrow the recording window and report the cost.
- If concurrency harms correctness, isolate the incompatible job class; if it only harms throughput, reduce admitted concurrency using measurements.
- If an apparent improvement disappears under independent replay or held-out testing, retain it as a failed hypothesis.

The aim remains a practical, testable account of model behavior. The stages are checkpoints for progress toward that aim, not permanent limits on it.
