# Model Looking Glass: initial research design

## Intent
Test Easton's proposed internal observation windows: preserve a model's activation vectors while recording where they were seen. Compare direct activation capture with a separate gradient reporting channel. The proposal follows a discussion of zero-forward-loss Trojan regularizers but does not import or run a Trojan payload.

## Research question
At equal probe locations and recorded tensor bytes, does transporting an activation through a separate gradient provide any observational or operational advantage over copying it directly? How does probe placement change the first visible location of an injected numerical perturbation?

## Scope and choices
This first repository uses Python/PyTorch, a small deterministic residual network, branch metadata, and synthetic inputs. These are implementation defaults, not claims of user-chosen hyperparameters. Direct hooks, gradient-reporting hooks, and ID-only hooks share locations and metadata. A no-probe baseline measures transparency and overhead. Sparse and dense placement are compared separately; recording more bytes is not presented as an algorithmic improvement.

## Channels
For an observed activation a, a forward hook returns None. Direct capture stores a detached clone. Gradient reporting creates a separate, ephemeral leaf q=0 and calculates grad_q sum(stopgrad(a)*q), giving a for finite values. Neither q nor the reporting computation is part of the model or optimizer. The hook does not replace the activation. This requires a local enabled-gradient context even during no_grad or inference_mode.

Identity taps expose named points inside the benchmark network. The transport comparison is between capture mechanisms at the same points, not between distinct access privileges. Event records carry run ID, node ID, invocation count, shape, dtype, and a detached value if requested. Identical-output branches demonstrate that values alone cannot identify a path; node IDs provide explicit instrumentation metadata.

## Verification
Tests compare outputs, scalar task loss, parameter gradients, input gradients, and one SGD update with identical initial states. Test repeated modules, cleanup on exceptions, disabled autograd/inference contexts, float32/float64 and supported bfloat16, plus non-finite input policy. Capture failures must not silently produce misleading reports. Monitoring storage is bounded by a configurable maximum number of events, with overflow explicitly rejected.

## Evaluation
Freeze a finite protocol before the first experiment: two widths, four seeds, two placements, three channels plus baseline, CPU eager float32. Measure deterministic equality and first-observed-divergence coverage, channel reconstruction, exact recorded tensor bytes, event counts, and median/p90 latency after warmup. Report platform, version, timing variability and small-workload limits. Peak process/GPU memory is not inferred from tensor bytes. No pretrained model, compiled execution, accelerator or safety-certification claim.

## Non-goals
No universal backdoor detector, inference about private models, claim to reconstruct reasoning, claim of novelty for standard hooks, or attempt to exploit rounding behavior. No remote telemetry, external dataset, model download, credentials or unrelated repository edits.

## Alternatives considered
1. Arithmetic cancellation gates: rejected because numerical ranges and backward behavior compromise reliable telemetry.
2. Direct activation hooks: baseline, simplest observation path.
3. Isolated gradient readout: explicit experimental channel matching the user's vector-through-gate idea; expected information equivalence is tested, not advertised as a gain.
