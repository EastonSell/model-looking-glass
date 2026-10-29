# Recorder CPU overhead protocol v1

This protocol characterizes the cost of `BatchedEventRecorder.record` in one
Python thread with synthetic metadata. It performs no neural-network forward,
model load or GPU operation. It is not a GPU throughput comparison, a device
latency benchmark, or a measurement of observer contention under concurrent
clients.

The fixed workload alternates two complete event templates: a `selected`
record and a two-member `forward_entry`. Both comparison modes construct the
same event dictionary and read `perf_counter_ns` for each event. In `discard`
mode a no-op sink returns success; in `record` mode the recorder validates and
copies each event under its lock. Recorder construction, snapshot/report and
artifact writing happen outside the timed window. Both payload classes are
warmed with 200 total events per mode; then seven paired blocks each time 400
events per mode. Four blocks are assigned discard-first and three record-first;
a fresh `Random(17)` shuffles that balanced allocation. The saved workload includes both complete templates, the seed,
and a canonical SHA-256 fingerprint.

Each raw sample records the common clock's window start, end and duration,
mode, repetition, event counts and recorder completeness. The saved summary
reports each mode's median window duration and each paired difference divided
by 400 events. Differences can be negative and are not clamped. A nominal clock
screen requires each complete window to be at least 100 times the recorded
`perf_counter` resolution and the clock must report monotonic, nonadjustable
behavior. The resolution is not a guarantee of accuracy,
scheduler timing, or isolation. Seven blocks on one machine are descriptive,
with no population or tail-latency claim. CPU frequency, host scheduling,
other process activity and execution order can affect the result. Python GC is
left enabled or disabled as observed; its state and thresholds, plus collection
and allocation-counter snapshots just outside each window, are saved. Those
snapshots can include boundary effects and do not isolate collections caused
solely by a timed window.

Run from a clean checkout after creating a new empty absolute bytecode prefix
and an output directory outside that checkout. For example, in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$recorderPrefix = '<new-empty-absolute-directory>'
& '<venv-python>' -B -X "pycache_prefix=$recorderPrefix" -m looking_glass.recorder_overhead --output '<new-absolute-result-path>'
```

The program requires an empty prefix when checked and disabled bytecode writes,
an empty CUDA visibility variable, and `torch.cuda.is_initialized() == False`
before and after the run. That last field is only PyTorch process-state
evidence. It checks that each project module still present in `sys.modules`
at either check matches the recorded Git commit and the checkout is clean
before and after timing; transient imports are not enumerated. The environment
and dependency directories, Python startup hooks and runtime libraries remain
trusted; source hashes do not attest executed memory, and concurrent source or
dependency changes are out of scope. The operator must actually create the
prefix empty before launching. The terminal JSON is published atomically with
no overwrite, after timing, and includes all 14 raw window samples. A failed
clock screen is saved with a nonzero exit status. An operational error or
interruption best-effort publishes a failed/cancelled terminal record with its
phase, earlier completed windows, and any available recorder-loss report;
it never invents a missing sample or summary. If publication itself fails or
the process is killed, no terminal artifact is guaranteed.

These numbers characterize the recorder operations only. Actual grouped and
serial paths emit different numbers and sizes of observations, contend for
locks, and may alter queue timing or batch formation. A future GPU experiment
must separately measure recorder-on versus minimally observed windows in
randomized paired blocks for **each** mode, report group-size and failure
changes, and apply matching memory and device controls. Do not mechanically
subtract this CPU difference from a GPU throughput or latency result.
