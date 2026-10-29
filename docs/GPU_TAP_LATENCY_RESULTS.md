# Two-token GPU tap latency: exploratory observations

On 2026-10-28, a supervised DistilGPT2 run collected six paired timing rounds
on an NVIDIA RTX 2070 SUPER. Its activity screen was **contaminated**, so these
are exploratory observations, not isolated-latency measurements or evidence of
a general slowdown. The run used one input, one capture location, and six
correlated rounds. It does not establish physical GPU or system-wide CPU
exclusivity.

| Mode | Median synchronized wall time | Median CUDA event span |
| --- | ---: | ---: |
| No capture | 21.5499 ms | 20.8591 ms |
| Direct capture | 22.1345 ms | 21.4230 ms |
| Auxiliary gradient readout | 23.0297 ms | 22.3172 ms |

The median of the six within-round gradient/direct wall-time ratios was
1.0392; the median direct/no-capture ratio was 1.0288. Individual ratios
varied substantially:

| Round | Execution order | No capture (ms) | Direct (ms) | Gradient (ms) | Gradient/direct | Direct/no capture |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | none, direct, gradient | 22.1203 | 22.7714 | 21.6982 | 0.9529 | 1.0294 |
| 1 | none, gradient, direct | 21.5612 | 20.9473 | 21.3323 | 1.0184 | 0.9715 |
| 2 | gradient, none, direct | 22.7050 | 23.0499 | 24.4068 | 1.0589 | 1.0152 |
| 3 | none, direct, gradient | 21.5386 | 22.1458 | 24.7510 | 1.1176 | 1.0282 |
| 4 | direct, gradient, none | 19.6906 | 21.9222 | 23.5034 | 1.0721 | 1.1133 |
| 5 | gradient, direct, none | 19.5651 | 22.1232 | 22.5560 | 1.0196 | 1.1307 |

These are descriptive medians of a small, correlated sample. Each measured
round contains all three modes in a prespecified shuffled order; two earlier
rounds supplied six warmup observations. Ratios are computed within rounds,
then summarized, rather than divided from the separately reported medians.

The workload was batch 1, tokens `[15496, 995]`, FP32/eager/eval DistilGPT2,
with one last-token capture at `transformer.h.0`'s block output. Each captured
activation had shape `1 x 1 x 768` and occupied 3,072 bytes. The gradient mode
uses a local auxiliary `autograd.grad` activation readout, including extra
cloning and reduction. It is not a full-model backward pass or a measurement
of model sensitivity. Both capture modes perform an activation-finiteness
reduction; every mode validates output logits. Consequently, direct/no-capture
includes capture and validation work as well as hook costs.

Wall times use the monotonic Windows `QueryPerformanceCounter` clock and include
the model call, validation and synchronization. CUDA events measure a span on
the checked default stream and do not necessarily include unrelated unjoined
streams. The equivalence preflight checked finite logits and equality under
`torch.equal` with no tolerance; that numerical comparator is not a guarantee
of identical stored bits, especially for signed zeros and backend-dependent
subnormal behavior.

The monitor retained 15 device observations with no query errors and complete
coverage under its declared rules. The NVIDIA process lists contained 17
distinct foreign PIDs across the endpoint queries. The three before-idle GPU
utilization samples were 1%, 1%, and 4%; the three after-idle samples were 19%,
3%, and 3%. Other listed processes are not necessarily actively computing, and
sampled utilization does not identify its source. WDDM endpoint process lists
and periodic polling can miss activity or overlap in time.

The process used a 25% PyTorch allocator cap, live headroom guards, and a
separately held device lease. The maximum PyTorch allocated/reserved peaks
observed across measured and warmup samples were 369,420,288 and 408,944,640
bytes. The composite equivalence preflight reached 370,630,656 allocated and
408,944,640 reserved bytes. These are observations, not device-wide peaks or
guaranteed upper bounds. The owner exited cleanly; after a 32 MiB retained
cuBLAS workspace was cleared and synchronized, current PyTorch allocated and
reserved bytes were both zero. No unsafe device marker remained.

## Earlier failed attempt

The first run completed the same number of warmup and measured calls but did
not produce an accepted timing result. All eight monitoring polls failed, and
both endpoint process queries returned 255. A Python 3.12 CPU suite also
likely overlapped the measured interval, based on an approximate mapping
between the high-resolution and wall clocks. The original terminal recorded
`failed` and the activity screen `unverified`; the owner itself exited and
cleaned up successfully. The missing Windows `PROGRAMFILES` variable was later
identified by a controlled read-only query as the cause of the NVML failure.
The first run remains failed and its timings must not be pooled with the retry.

## Reproducibility and public evidence

The [retry source snapshot](https://github.com/EastonSell/model-looking-glass/tree/616ff2d9bcc274587c650cdbb8e7a643f5f47a5f)
is the exact revision recorded in the run. The first run recorded local commit
`151b18e778176c651ceb0047aaabeb5670b86d99`; its
[published source snapshot](https://github.com/EastonSell/model-looking-glass/tree/d5dfcc4e301c3a90ca41c4a57ea7be30fc23b9c6)
has the identical Git tree `b7792db6c6f92166532f48117d71a7c794703cf7` but a
different commit ID. Both used pinned model revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, PyTorch 2.8.0+cu128, and
Transformers 4.57.3.

The [failed-attempt evidence](../evidence/gpu-tap-latency/v1/) and
[exploratory-retry evidence](../evidence/gpu-tap-latency/v2/) contain the
original warmup, measured and postcheck JSONL records, CPU preflight, public
activity-screen and telemetry projections, cleanup snapshots, and a summary.
Copied original files were checked against the respective sealed run inventory;
each public directory has its own `subset_inventory.json`. The full private
inventories and private process/device identity records remain with the original
local runs and are not represented as fully replayable from the public subset.
The public projections omit physical GPU identifiers, PIDs and process names,
so a public reader cannot independently reconstruct the private identity or PID
comparisons. Independent local audits checked those bindings, raw-row arithmetic,
screen classification, source and asset hashes, and cleanup before publication.
