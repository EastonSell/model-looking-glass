# Two-token direct-versus-gradient tap timing, development protocol v1

Status: **proposed and not run**. This is a small isolated timing experiment,
separate from the shared-model throughput work and the eight-token benchmark.
It compares implementations at one selected transformer boundary; observing
an activation does not itself explain a model's behavior.

Use the verified offline `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, float32 eager evaluation, no KV
cache and `torch.no_grad()`. The fixed batch is `[15496, 995]`. Capture only
the layer-0 block output at batch 0, token 1, on its source device. Direct and
gradient sessions retain one `(1, 1, 768)` tensor (3,072 bytes); cap each
session at two events and 64 KiB. The caps do not include temporary tensors,
allocator rounding, model weights or CUDA workspace.

Before timing, compare the baseline and the ID/direct/gradient outputs and
capture coordinates. Require finite logits and `torch.equal` on outputs and
selected direct/gradient values with matching shape, dtype and device. This
is numerical equality without an explicit tolerance, not a bitwise guarantee:
signed zero compares equal and subnormal handling can vary by backend.

The process must use the common UUID-derived device lease and supervisor,
verify the actual CUDA UUID, limit the PyTorch allocator to 25% of device
memory, and read back its effective cap. Require at least 3 GiB device-free
before model transfer. Before and after the preflight and each sample, check
the lease, device-free reserve of `max(1 GiB, 15% of total) + 256 MiB`, and
the 256 MiB margin below the allocator cap. Synchronized preflight and warmup
peaks measure the actual one-block capture path before measured rounds; abort
on pressure or OOM instead of changing the workload. These are bounded,
observed samples, not a proof of a worst-case upper bound. The earlier
two-token no-capture pilot does not itself authorize this capture workload.
The composite equivalence preflight retains both direct and gradient traces
and multiple outputs simultaneously; its observed peak covers that actual
sequence. Synchronize while the final output, baseline and both captured
sessions are still alive, then synchronize again after their disposal. Preserve
a provisional whole-run memory commitment and both trace
allowances until the child has disposed of device state, synchronized and
exited. The allocator cap and free-memory checks do not reserve physical VRAM.

Use seed 17. Interleave `none`, `direct`, and `gradient` once per round: two
warmup rounds shuffled with seed `17 XOR 0x5A17`, followed by six measured
rounds shuffled with seed 17. Save every completed raw row and ordering;
exclude warmups from summaries. The prespecified summary is the median
within-round gradient/direct wall-time ratio, with per-mode wall medians.
Also report the median within-round direct/none ratio.
These are correlated repetitions of one model and input, not independent
examples or a general hardware speedup claim.

Synchronize before each wall interval, use events explicitly on the checked
default CUDA stream, and synchronize with outputs and traces alive before
disposal, then synchronize again. Wall intervals and all telemetry timestamps
use `time.perf_counter_ns()` in one clock domain; record its implementation
and resolution, and require a monotonic clock with at most 1 microsecond
resolution. Require each measured or warmup wall
interval to span at least 100 reported resolution units; otherwise abort.
The event span is reported separately. Each mode includes the same finite-logit
reduction and host synchronization; instrumented modes also include hook
registration/removal, trace validation, the selected capture and an extra
selected-activation finiteness reduction. Thus direct/none includes capture
and validation as well as hook overhead. Gradient capture includes an
additional clone and isolated autograd readout. CUDA
events can include host-induced gaps and omit transient unjoined custom-stream
work; synchronized wall time is the complete reported interval. Record
process allocator baselines and peaks separately from sampled device-free
snapshots, which are not device-wide peak memory. Journal writes and fsync
occur between timed samples and can affect subsequent system state.

Use a distinct new private run directory, durable start/partial rows and a
terminal result published only after the exact supervised child is reaped and
its CUDA teardown verified. Record source revision, checkpoint asset hashes,
device/driver and software versions, allocator cap/readback, seed/order,
checks, raw samples, and failure/OOM/cleanup status. Screen and disclose
external GPU activity using the idle, coverage and process-query rules in
`CUDA_BENCHMARK_V1.md` and its classifier. Mark the screen unverified when
queries fail, coverage is insufficient or device identity is ambiguous; mark
observed foreign activity as contaminated. WDDM and periodic samples cannot
prove physical exclusivity. The cooperative lease does not control other desktop
applications. Each recorded preflight or sample observation starts as pending;
only a separate matching passed lease/memory postcheck validates it for a
successful summary. An interrupted observation remains visibly pending. Do not
publish private paths or physical GPU identifiers. This
protocol requires independent CPU review and a separately scheduled GPU
window before execution.

Start `looking_glass/gpu_tap_latency_entry.py` directly with Python 3.11 or
later, `-I -B -S` and an absolute, nonexistent `-X pycache_prefix`. Supply
the explicit trusted `--site-packages`, shared `--workspace-root`, offline
`--cache-dir`, `--smi`, new private `--output` directory and `--exclusive`
only for an actually coordinated window. The controller checks source and
asset bytes before launching the supervised child. The local run directory
contains private UUID and telemetry files; any public export needs a separate
filtered inventory and privacy review. A failed owner keeps its process-local
custody and the device lease until process exit; a failed or ambiguous drain
never receives a clean attestation. Arbitrarily interrupted interpreter return
temporaries cannot all be rooted, so a quarantined owner must promptly exit.
