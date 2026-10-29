# GPU tap timing: one contaminated v1.3 session

This is a completed **exploratory, contaminated** session, not an isolated benchmark or a general estimate of tap overhead. The activity screen was complete, but listed 17 foreign process IDs and recorded 11% GPU utilization in each of the three selected observations both before and after the workload. A listed process is not proof of its GPU utilization. The fixed protocol classifies either finding as contamination.

The session used DistilGPT2, one fixed two-token input, batch size one, float32 eager attention, and a single NVIDIA GeForce RTX 2070 SUPER. It ran one preflight, six warmups and 36 measured samples in 12 balanced three-mode rounds. Direct controller exit 0 was separately observed. The saved owner final reports synchronized disposal, released claims, and zero PyTorch allocated and reserved bytes. The successful source path requires the same owner's clean zero-exit supervision and Job retirement; no independent Job lifetime-count receipt was saved, and these records do not prove native CUDA cleanup or global idleness.

The frozen source is commit `0c14778447ed6dcbc4890984641370659a09ffef` (tree `0ff3b037d19a5bc839e634524d480b6eace5d7c2`), protocol `gpu-tap-latency-repeated-v1.3`. The model revision, fixed input, runtime settings, CUDA build and driver point observation are in `public-binding.json`. This session is separate from the two incomplete v1.2 attempts and is not pooled with them.

## Descriptive observations

| Mode | Median primary wall interval | Median CUDA event interval | Median guarded interval |
| --- | ---: | ---: | ---: |
| no tap | 45.9541 ms | 44.9953 ms | 5.2849 s |
| direct tap | 45.5441 ms | 44.6334 ms | 4.7840 s |
| gradient tap | 51.2466 ms | 50.2264 ms | 4.9098 s |

Across the 12 paired rounds, the median of gradient/direct wall ratios was 1.0679 and the median of direct/no-tap wall ratios was 1.0908. These are medians of paired ratios, so they need not equal ratios of the mode medians in the table. Individual paired ratios vary substantially; this one contaminated session supports no general speed or overhead claim. The guarded interval includes checks and disposal outside the primary interval; it is not another measurement of the same latency. Parent telemetry and fsync may overlap wall time.

The largest recorded reset PyTorch peaks were 370,630,656 bytes allocated and 408,944,640 bytes reserved. The smallest CUDA free-memory point among the saved admission, pre-transfer and journal snapshots was 7,016,022,016 bytes. These are sampled/process-local observations, not a whole-device peak, native cleanup proof, or a bound for a future workload.

## Host context

Windows system CPU counters bracketed the controller lifetime from 10:28:59 to 10:34:01 UTC and imply a 59.65% aggregate busy fraction over that bracket. It includes the timing controller and observer; on hosts with more than 64 logical processors the API may cover only the calling thread's processor group. A CUDA-hidden CPU test-suite process point and a separate CPU study interval overlap that controller bracket. Other CPU-only tests and checkpoint hashing were reported during approximately the same period. The records do not establish per-sample overlap or the cause of any timing variation. `host-context.json` distinguishes retained records from approximate operator reports.

## Evidence and replay

- `session-result.json` is the path-free controller result, including the public activity sample series and paired statistics.
- `journal.jsonl` contains all 86 saved observation and postcheck rows; `schedule.json` is the frozen schedule.
- `audit-timing.json` is an independent saved-record timing replay. `audit-screen-host.json` and `audit-screen-independent.json` are separate reviews of the private activity and host records. Their hash commitments permit correlation with retained private evidence; raw process IDs, device UUIDs and local paths are not published.
- `public-binding.json` projects fixed model/runtime/device settings from the private candidate, and `host-context.json` records the overlap scope.
- From this repository checkout, `python -I -S -B results/gpu-tap-latency-v13-exploratory-20261029/replay_public.py` verifies the pinned protocol/schedule/journal bytes and recomputes the public paired timing summary. It does not validate the private process endpoints, lifecycle or authenticity of telemetry. The independent full reviews used retained private records.

The observer required a separately observed zero controller exit and no incomplete sidecar. Its retained receipt and the post-exit OS point are private; their hash commitments appear in the independent audit. A missing separately persisted Job inventory and the limitations of sampled GPU/CPU observations remain explicit.
