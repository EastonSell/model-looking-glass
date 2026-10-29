# Shared-model throughput protocol v1

Status: CPU measurement machinery implemented; supervised CUDA integration under
review. This protocol is separate
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
confidence interval. With eight requests per full window, and four per team,
nearest-rank p95 is the maximum; it is not a useful SLA or stable tail estimate.

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
it is Jain's index over each team's succeeded/attempted fractions, rather than
absolute completions or service time. The fixed offered workload, unequal team
costs and observation horizon matter. In serial mode all team-a requests precede
team-b requests; the latter team's time before it first submits is absent from
attempt and queue timings. The modes do not provide equivalent arrival traces.

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
Output tensors are not retained: an offline reader can compare the stored digest
strings, but cannot independently recompute the original full-output hashes or
finiteness checks. Replay is a separate execution, not proof of the original.

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

The CUDA controller first writes an initializing record and verifies pinned cache
assets and fixed inputs on the CPU. It requires a clean committed source tree,
checks each benchmark source hash against the corresponding Git commit blob, and
requires an output directory outside the checkout. The supported GPU entry point
is the directly executed source bootstrap below. Before importing project modules,
it requires `-I -S -B`, explicitly selected dependency directories, and a new
empty absolute `-X pycache_prefix`; it rejects prior project imports. The
supervised child gets a separate empty prefix and the same dependency directories.
`-B` alone still reads existing bytecode caches. Python's hash randomization is
enabled but its process seed is not fixed because `-I` ignores Python environment
variables; the protocol and request seeds are recorded separately. The bootstrap
loads a source allowlist for the project package without adding the checkout root
ahead of the standard library. Operator-selected dependency directories are trusted,
not authenticated or fully content-hashed; software versions do not prove binary
identity. Do not modify the checkout or those directories while a run is starting
or active: filesystem hashes are observations, not in-memory code attestation.
The same bootstrap can launch a CPU run; its setup records the validated startup
status. A normal `python -m` CPU run records bytecode isolation as unverified.
CPU preflight temporarily masks all visible GPUs. The controller identifies the physical GPU from a
trusted `nvidia-smi` UUID inventory and launches the real Python interpreter as
the supervisor's direct child, including on Windows virtual environments. All
cooperating owners must use the same absolute device coordination root. Select a
full physical GPU UUID through `CUDA_VISIBLE_DEVICES`, or leave it unset only on
a host with one physical GPU. The child acquires the UUID-derived device lease,
checks the supervisor's sticky marker, and verifies the runtime UUID and visible
device count before model work. It keeps the lease until process exit.

Before importing torch the controller and owner reject CUDA launch blocking,
allocator, cuDNN and TF32 environment overrides, and unrecognized CUDA/PyTorch
or CPU threading environment keys. They set lazy CUDA module loading, one thread
for OMP, MKL, OpenBLAS, NumExpr, BLIS and vecLib, and a deterministic cuBLAS
workspace, while recording the permitted device visibility,
ordering, cache and connection settings. After verifying the physical device the
child checks the native allocator backend, sets its per-process PyTorch allocator
fraction to 0.75, reads it back, and records deterministic and TF32 settings.
This cap covers PyTorch's caching allocator, not the CUDA context, other device
allocations or other processes.
The policy also rejects toolkit-path overrides such as `CUDA_PATH` and
`CUDA_PATH_V12_0` if they are inherited from the host. Remove unsupported keys
from the launch process environment before running; the failure record lists the
key names and does not save their values. The supervisor starts the owner with
an exact environment containing the validated runtime settings and selected OS
path, profile and temporary-directory variables; it does not inherit other
parent variables. Those selected OS values and the installed runtime remain
trusted operational inputs rather than authenticated binaries.

The child also holds a 2 GiB model-load estimate and separately admits each
fixed-shape forward with a 1 GiB estimate, the budget's reserve, and zero GPU and
host trace caps. These are provisional, unmeasured estimates, not hard allocation
limits or proof that OOM cannot occur. Normal GPU admission is disabled until a
reviewed measured profile matches this exact workload and runtime. The explicit
`--development-unprofiled` flag permits only a coordinated `--quick` development run;
even a clean, internally valid run is reported as `development_only` by the
controller and its command exits nonzero. Its records cannot establish a normal
admission profile or authorize other workloads. Owner records, raw attempts and
measurement artifacts also carry the unprofiled development label, while their
internal completion statuses only describe execution and cleanup. Device-wide free memory and process allocator
counters are recorded. The broker uses one resident model and execution thread,
four pending slots and one pending slot per team; it neither batches nor overlaps
model forwards. External applications can still allocate GPU memory.

After broker drain and synchronization the owner clears the PyTorch allocator
cache and checks that both allocated and reserved tensor bytes are zero before
attesting clean. A timeout, invalid physical binding, retained allocation, failed
synchronization or uncertain drain leaves the supervisor's unsafe marker in place.
The controller accepts a result only after a clean reaped child with exit code
zero and complete matching source, workload, raw attempts, timings, memory,
comparison and cleanup records. A scientific failure can have a clean hardware
handoff, but remains a failed experiment. GPU evidence contains device UUIDs,
inventory and possibly local paths in failure messages: keep it private and
review/redact any copy before publication. For example, after reserving a shared
GPU window and substituting the shared root, full UUID and private output path:

```powershell
$cachePrefix = Join-Path $env:TEMP ("mlg-throughput-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $cachePrefix | Out-Null
python -I -S -B -X "pycache_prefix=$cachePrefix" scripts\throughput_bootstrap.py `
  --bootstrap-site C:\path\to\venv\Lib\site-packages `
  --output C:\path\to\private-output --cache-dir C:\path\to\hf-cache\hub `
  --device cuda:0 --device-root C:\path\to\shared-device-root `
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --quick --development-unprofiled
```

The fixed v1 driver permits at most ten repetitions. Its JSONL and JSON reads
are bounded; use a new run directory for additional experiments.

Three repeated windows over four duplicated public prompts are correlated,
descriptive observations. A rate change can include overlapped client-side CPU
hashing or validation and queue effects; it does not establish GPU kernel overlap.

The CPU driver writes `setup.json` and `measurements/` under its run directory.
The GPU controller writes `controller_manifest.json` before preflight and
`preflight.json` after it succeeds. Its supervised child writes `owner/setup.json`,
`owner/admission_profiles.json` and `owner/measurements/`; child `failure.json`,
`cleanup.json`, and `run_status.json` record setup and teardown. The controller's
terminal `controller_status.json` includes the reaped process health and hashes
of the owner evidence, initial controller manifest and validated CPU preflight.
`controller_failure.json` preserves a controller exception and hashes the parent
records that were successfully written;
an initializing record without a terminal record is incomplete. Measurement and
cleanup can disagree, so nested results alone do not establish successful device
handoff. A failed or
uncertain drain retains the worker/lease, marks cleanup failed, and requires
terminating the supervised process rather than reusing its CUDA context. Both
CPU and GPU measurements include CPU-side finite checks and full-result hashing,
so rates describe this validated service workload rather than raw model kernels.
