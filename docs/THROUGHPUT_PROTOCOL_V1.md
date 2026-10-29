# Shared-model throughput protocol v1

Status: implemented measurement machinery under review. This protocol is separate
from isolated latency measurements and the initial CPU experiment. No GPU speedup,
causal explanation, batching, CUDA overlap, or fairness guarantee follows from it.

## Question and workload

For one resident model and the same fixed set of requests, compare a single
closed-loop serial client with one concurrent closed-loop client per team. Each
client waits for a terminal response before submitting its next request. The
broker has one execution owner: simultaneous submissions mean queued work, not
simultaneous GPU kernels. The modes offer different load; this is a load or
saturation comparison, not the same arrival trace and not isolated latency.

Before running, specify model/checkpoint and tokenizer IDs, immutable revisions
and hashes, model class, dtype, device, evaluation/cache/backend settings, source
commit and dirty state, all software/driver versions, team assignments, request
order, capture/intervention policy, input generation and seeds, fixed work-unit
definition, queue/worker/output bounds, memory policy, repetitions, and external
GPU activity. Keep these choices in the run manifest. The concrete driver must
validate that token units equal the counted input positions and say whether
padding is included. Generic API units are **caller-declared** and are not a
verified token count. Use public or otherwise permitted inputs; persisted evidence
contains hashes and metadata, not input or output tensors.

The normalized manifest hashes all fields of each supported request, including
input tensor shape, dtype and SHA-256 over its contiguous bytes, plus job/team,
seed, class, declared units, policy, and the caller-supplied model specification.
Unsupported request field types fail closed. The caller must prevent concurrent
mutation of requests and their underlying tensors throughout every window. The
harness rechecks the frozen manifest before/after each window and hashes each
request at submission; these checks detect drift but cannot eliminate a transient
race with an uncooperative writer. Model and tokenizer metadata supplied
by a caller are not verified by the generic API; a concrete driver must bind them
to a verified checkpoint. A full-result SHA-256 over canonical metadata and
bit-exact CPU tensor values is the concrete driver's exact equality criterion.
A generic digest callback defines only the fields it actually examines and cannot
prove equivalence of omitted outputs or traces. Different results, failed jobs,
rejections, unresolved outcomes, different manifests, or an overlapping/same-mode
pair invalidate a paired throughput comparison. Invalid runs retain all evidence.

## Execution and timing

Use a single warmed resident model for both modes. Before each measured window,
run every exact request once per configured warmup round in that mode; a failed
warmup aborts. For each repetition, choose serial/shared order using the recorded
protocol seed. Within each team the manifest order is fixed. Client scheduling
and actual cross-team arrival order are not deterministic; save all attempt times.
Request fingerprinting and output validation/digest work are included in client
response and the overall window. Artifact writes occur outside the measurement window.

All timestamps use `time.perf_counter_ns()`, the same clock as the broker. Save
its implementation and nominal resolution. On the inspected Windows Python 3.11
host this is QueryPerformanceCounter (nominal 100 ns); `monotonic_ns()` is
GetTickCount64 with nominal 15.625 ms and is unsuitable for this workload. Actual
resolution, timer overhead, scheduling and tail uncertainty still matter. A
barrier action stamps the beginning immediately before clients are released; the
end is after all client threads have received and validated their terminal
outcomes. Successful jobs or caller-declared work units divided by that whole
interval are the two throughput metrics. Startup/model load and warmups are
outside the window; all queueing, broker work, client validation and final thread
drain inside it count. Per-team rates use this same global window denominator.

The broker records `submitted_ns` at submit entry, `ready_ns` after the input
snapshot is ready for dispatch, `started_ns` at dispatch, and `finished_ns` at
terminal outcome. Its queue slot is reserved before the snapshot while the owner
cannot dequeue it. Thus ready-to-start is the reported queue wait; submitted-to-
start additionally includes lock contention and snapshot work. Service is finish
minus start, including the synchronized worker, result CPU copy and cleanup.
Broker response is finish minus submitted; caller response is returned minus
attempt and includes validation. These spans have different boundaries. Report
p50 (midpoint median), p95 (nearest rank), maximum and count by terminal state as
well as overall and per team. Immediate rejections must not be silently mixed into
a success-only latency distribution. Small-sample p95 is descriptive, not a
confidence interval.

For CUDA the benchmark proxy synchronizes the target device in the actual worker
thread before and after every execution, including failure/cancellation cleanup.
A synchronization failure poisons the proxy and prevents worker reuse. This is
end-to-end wall time, not a device-only CUDA-event span. It does not establish
that unrelated processes are absent, or that kernels overlap. Before/after
snapshots distinguish device-wide free/total from this process's PyTorch
allocated/reserved bytes. After warming, reset process allocator peak counters;
report peak allocated and reserved over each measurement separately. Device-wide
free values are boundary snapshots, not a device-wide peak. Driver allocations,
other processes and WDDM usage can be outside PyTorch allocator statistics.

## Failures, fairness, and evidence

Report attempted, accepted into the broker, rejected, succeeded, failed,
cancelled, unresolved, invalid-output, and unexpected-submit counts, with raw
errors and timestamps by team and class. Accepted is a queue-acceptance count,
not proof of separate GPU memory admission. Completed jobs count toward useful
throughput only after output validation. Report each team's counts, work, queue
wait and response/service distributions and maximum. The completion-balance Jain
index is descriptive: with full draining and no failures it becomes 1 even if a
team waited much longer. It is not evidence of time fairness or non-starvation;
the fixed offered workload, unequal team costs and observation horizon matter.

An unresolved wait or unexpected submit signals all clients to stop and requests
cancellation of known handles; a concurrently admitted late handle is cancelled
when it is registered. A cancellation
request is cooperative, not proof the kernel stopped. Drain or close the broker
and verify completion before restarting; if CUDA synchronization fails, quarantine
the worker/process. Wait and cancellation timeouts do **not** bound blocked
submission, hashing, client startup or thread joins. A hard wall-clock limit needs
a separately supervised process; Python threads cannot safely be force-killed.
Neither an OOM nor a changed budget silently changes the workload or makes the
run successful.

Each run creates a new directory. `manifest.json` is published before the first
job; `warmup_samples.jsonl`, `raw_samples.jsonl`, and `partial_windows.jsonl`
record the attempts and completed windows. `results.json` is atomically
published only after all windows, and marks an unequal/failed pair
`invalid_comparison`. `failure.json` identifies failed or cancelled execution.
Interrupted windows preserve completed samples, known active handles and
submissions with unknown acceptance separately, and state that work may still be
running. Broker UUID plus broker job ID and benchmark session ID allow correlation.
No existing evidence directory or terminal JSON is overwritten. JSONL appends
are flushed but are not an atomic multi-record transaction; a hard process crash
can leave a partial last line. Preserve the directory, treat an unparseable or
missing terminal record as incomplete, and start a new run after recovery. This
local writer records SHA-256 hashes of its evidence files but is not the experiment registry and does not provide its transaction
or verification guarantees. Compare stored file hashes when supplied; they are
integrity checks, not cryptographic signatures.

The protocol does not infer reasoning from activation traces. In the current
forward-only worker it captures no traces at all. Direct-versus-gradient tap
comparisons belong to the separate controlled instrumentation protocol, with
matched observation points and independently verified outputs.


## Concrete DistilGPT2 driver

The standalone entry point uses the verified, offline-first DistilGPT2 loader
(`distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`,
with pinned asset hashes). It accepts an explicit Hugging Face cache directory;
it does not download assets. The full workload uses the four public prompts
in `DISTILGPT2_PROMPTS`, duplicated across teams `team-a` and `team-b`, with
seed `1000 + prompt index`, batch one, no padding or special tokens, float32,
eager attention, eval mode, no generation cache, and no capture or intervention.
The worker bounds sequence length at 128, input bytes at 1 MiB and output bytes
at 8 MiB. Units are derived from the actual input tensor element count; the
reported unit is processed input positions including any padding (this workload
has none). Full-request hashing, output validation (finite float32 rank-three logits
and expected vocabulary), and exact full-result hashing run on the CPU. The worker
also validates each result's batch/sequence dimensions.

The full protocol uses three repetitions by default and warms every job before
each mode. `--quick` instead runs two prompts per team and one repetition: it
is a functionality smoke check and cannot substantiate a speedup. Example:

```powershell
python -m looking_glass.throughput_benchmark `
  --output results/my-throughput-cpu --cache-dir C:\path\to\hf-cache\hub `
  --device cpu --quick
```

GPU execution is currently disabled: the driver must first integrate the reviewed
cross-process supervisor and physical-device identity binding. The following CUDA
setup describes the intended protocol, not a currently runnable command. Once
that gate is implemented, the coordinator must reserve a shared-service window.
The driver requires a
fresh CUDA process, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization when unset, rejects unsupported values, and records the setting.
The CUDA driver obtains one process-local 2 GiB estimated memory lease before
moving the model, leaving the budget's device reserve and requiring zero GPU or
host trace bytes. This is an admission estimate for the fixed forward workload,
not a hard runtime allocation limit, per-job re-admission or system-wide GPU
lock. Device-wide free memory and PyTorch allocator counters are still recorded;
external allocations can change after admission and cause OOM. The broker uses a
single model owner and one execution thread with four pending slots and one
pending slot per team. It does not batch or overlap model forwards.

`setup.json` records checkpoint/source identifiers, hashes, environment and the
replay descriptor before loading the model. The nested `measurements/` directory
holds protocol evidence. Root `failure.json` reports setup/driver failures and
`cleanup.json` reports broker and memory-lease cleanup separately, and
`run_status.json` is published after cleanup. A run is operationally successful
only when measurement and cleanup statuses are both `complete`; nested results
can exist even when cleanup subsequently fails. A failed or
uncertain drain retains the worker/lease, marks cleanup failed, and requires
terminating the supervised process rather than reusing its CUDA context. Both
CPU and GPU measurements include CPU-side finite checks and full-result hashing,
so rates describe this validated service workload rather than raw model kernels.
