# Initial results — 2026-10-28

## Bottom line

Transparent taps provide useful, explicitly named observation points. In this experiment the isolated-gradient and direct recordings compared equal under the recorded floating-point check, but the gradient channel was slower. There is **no measured information advantage** to that channel. Dense placement improves observation granularity at the cost of more events, retained values and latency.

This is a small synthetic CPU experiment, **not a pretrained-model study, a Trojan detector or a safety guarantee**. The identity itself is standard autodiff; no novelty claim is made for hooks or that identity.

## Verification

The final recorded run is [verified-cpu-v1](results/verified-cpu-v1/results.json), measured 19:18:39–19:18:45 UTC on 2026-10-28. Source hashes are included in its JSON. The measured implementation is local commit `68d17069a67afc2a23df28d72c9b3e9b64811b09`; publication may use a different commit because the GitHub connector publishes a content snapshot.

- **48/48 training-transparency cases:** tensor checks using `torch.equal` passed for outputs, scalar loss, input gradients, parameter gradients and state after one SGD update; matching absent parameter gradients were accepted.
- **16/16 channel comparisons:** `torch.equal` passed across **51,200 recorded activation elements**.
- **Identical-branch negative control:** equal outputs and activation streams, but distinct explicit branch labels. Vector contents alone do not identify unique execution history.
- **192 perturbation trials:** dense and sparse each detected 96/96. Dense first reported the injected tap in 96/96; sparse first reported a later block output in 96/96.
- **26/26 unit tests passed**, both normally and under `python -O`. Independent review includes separate execution, packaging verification, raw-CSV checks and correction evidence. See [REVIEW.md](REVIEW.md) and [verification status](evidence/STATUS.md).

Perturbations were deliberately inserted immediately before known taps. Thus exact dense localization is an instrumentation sanity check, **not evidence that the method identifies arbitrary hidden bug causes**. The reported interval is the last equal / first unequal observation, not a causal explanation.

**Later numerical qualification:** the run did not record denormal mode or save
the activation and control tensor bits. A separate [CPU control](docs/GRADIENT_DENORMAL_V1.md)
shows that under denormal flushing `torch.equal` can report equality when a
gradient readout has lost subnormal bits. The original results and raw evidence
are preserved; their checks do not establish bitwise equality or tell us whether
that edge case occurred in the historical run.

## Latency and recorded values

Each latency below is the median of four seed-specific medians, in microseconds per complete forward call. Each seed-specific cell has 25 warmups and 100 measured calls. CPU eager float32, one PyTorch intra-op thread, batch 8, depth 4, Python 3.12.14, PyTorch 2.15.0+cpu. Cells run in deterministic shuffled order. There was no known concurrent test workload during this final measurement; the host was not otherwise exclusively reserved.

| Width | Placement | ID only, µs | Direct, µs | Gradient, µs | Gradient / direct¹ | Events | Recorded value bytes² |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | sparse | 119.40 | 170.91 | 483.57 | 2.79× | 4 | 2,048 |
| 16 | dense | 178.12 | 361.47 | 1,376.68 | 3.81× | 16 | 8,192 |
| 64 | sparse | 135.45 | 193.46 | 506.93 | 2.60× | 4 | 8,192 |
| 64 | dense | 191.57 | 390.98 | 1,420.68 | 3.64× | 16 | 32,768 |

Uninstrumented baseline: **72.03 µs** at width 16; **85.78 µs** at width 64.

¹ Median of the four paired, per-seed gradient/direct ratios, not a ratio of the displayed aggregate medians.

² Direct and gradient retain the same number of value bytes. ID-only retains zero tensor-value bytes, but still uses metadata memory. This metric excludes metadata, intermediate tensors and allocators; it is **not peak memory**.

Timing includes session registration, forward execution, finite-value checks, capture, metadata, recorded-byte calculation, cleanup and disposal. It does **not** time task backward or optimizer updates. These are descriptive measurements of this workload, not confidence intervals, a universal slowdown factor or evidence about LLM-scale deployment.

## Evidence and reproduction

- [Raw JSON](results/verified-cpu-v1/results.json), [5,600 raw CSV timing samples](results/verified-cpu-v1/raw_timing.csv), [56-cell CSV summary](results/verified-cpu-v1/timing.csv).
- [Final run log](evidence/verified-full-run.txt); [normal tests](evidence/tests-final.txt); [optimized tests](evidence/tests-final-optimized.txt).
- [Frozen protocol](PROTOCOL.md); [implementation clarifications](docs/PROTOCOL_NOTES.md); [tested package versions](requirements-tested.txt).
- An [earlier full run](results/initial-cpu-v1/results.json) is preserved, with its own source hashes. It is not pooled into the table above and showed the same numerical conclusions. The final run follows additional failure-handling and autocast guards.

Run from a source checkout with the documented dependency installed:

```bash
python -m unittest discover -s tests -v
python -m looking_glass.experiment --output results/my-full-run
```

The next informative study would vary real model architecture, tap placement and storage budget; it should not presume a benefit from converting an already-readable activation into a gradient. No such follow-on results are claimed here.
