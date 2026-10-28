# Protocol clarifications and run history

Protocol v1's hypotheses, seeds, model sizes, perturbations, placement, sample counts and numerical acceptance criteria remained unchanged. The canonical `PROTOCOL.md` is byte-identical to the packaged copy.

The following implementation corrections enforce that protocol rather than changing its hypotheses:

- Construct model parameters and input tensors explicitly on CPU as float32, regardless of ambient PyTorch defaults.
- Reject active CPU autocast instead of labeling mixed-precision measurements float32. The standalone CLI uses ordinary grad mode; embedding it under inference/no-grad contexts is not the supported experiment entry point.
- Preserve raw timing CSV as well as raw JSON and summary CSV.
- Bundle the protocol in installed packages so the recorded hash does not depend on a source checkout.
- Record failures after import/output-directory creation. Missing dependencies fail earlier on stderr, which should be retained separately. If writing `failure.json` itself fails, the original evaluation exception is raised with the recording error chained as its cause.

An initial full run at `results/initial-cpu-v1` used source commit `38ff944e46f05ec06a58e07b2905f8b6bff6c4a3`. Independent tests occurred after that run, not concurrently. It is preserved rather than overwritten.

A fresh final run at `results/verified-cpu-v1` used source commit `68d17069a67afc2a23df28d72c9b3e9b64811b09`, after the autocast/failure-evidence regressions were fixed. It ran 2026-10-28 19:18:39–19:18:45 UTC, with the independent reviewer idle. This final run alone supplies the published latency table; no best-of-run selection or pooling was used. Both runs reached the same numerical conclusions. Absolute timings can vary with host load; the machine was not exclusively reserved.

The standalone runner changes PyTorch's RNG seed and intra-op thread count in its process. It is a research CLI, not a side-effect-free benchmark library. `TraceSession` itself does not change either setting. Repeated-module occurrences are tracked by the API, but this experiment's first-difference helper reports node labels, not a general repeated-node causal path.
