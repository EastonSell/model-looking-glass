# Frozen initial protocol

Protocol version: 1, specified before any complete experiment.

## Hypotheses
H1: Direct and isolated-gradient capture return equal detached activation tensors at equal locations, without changing task outputs, loss, input/model gradients or one SGD update.
H2: An identical-output branch pair is indistinguishable by values at a downstream merge. Explicit executed-node labels distinguish the branch when a branch-local tap is present.
H3: Dense observation points narrow the first-observed-divergence interval compared with block-output-only taps, but may increase event counts and recorded bytes.
H4: The extra gradient computation has a measurable overhead; there is no prespecified claim of a speed or memory advantage.

## Fixed configuration
- Seeds: 0, 1, 2, 3.
- Widths: 16 and 64; depth: 4; batch size: 8.
- CPU eager float32; one PyTorch intra-op thread.
- Channels: none (baseline), direct, gradient, id.
- Placement: sparse block outputs and dense all tap points.
- Twenty-five warmup iterations and 100 measured iterations per timing cell, in deterministic shuffled cell order.
- One task forward, squared-output mean loss, backward, and SGD update (learning rate 0.01) for transparency checks.
- Each block routes through branch 0 or 1, controlled by an explicit integer route input. Both branches are evaluated separately in diagnostic checks, not batch-dependent routing.
- Known perturbation: add 0.125 to the first flattened element immediately before each of 4*3 named intra-block taps, one perturbation per run. The interval between the last equal and first unequal observed tap is the output, not inferred causal explanation.
- Separate identical-branch negative control: copy branch weights, use equal inputs, compare downstream vectors and branch-local metadata.

## Required outputs
Raw JSON/CSV timing rows, summary, deterministic verification results, environment versions, protocol/code hashes and fixed seeds. Wall-clock measurements are descriptive for this CPU and small model. No inferential speed claim, benchmark leaderboard, general model-quality or safety conclusion.

Recorded tensor bytes are exact sum(numel*element_size) over saved values. They exclude metadata and temporary allocations and must not be labeled peak memory. No GPU peak memory result is planned for this initial CPU run.

## Failure handling
No overwriting an existing result directory. Fail on any transparency mismatch, nonfinite capture, unsupported configuration or missing dependency. Retain failure evidence. Do not silently skip a channel or substitute results. Any protocol change is versioned and documented before a fresh run; failed old output remains preserved.
