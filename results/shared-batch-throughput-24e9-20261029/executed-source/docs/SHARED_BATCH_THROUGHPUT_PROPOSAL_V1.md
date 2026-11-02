# Matched shared-model throughput: prospective first comparison

This is a prospective small experiment, not a GPU admission. It compares two clients sending the same planned
requests through a single resident model, first as serial single-row work and
then as compatible two-row work. The direct-versus-gradient latency study is
separate and its measurements must not be pooled with this experiment.

The existing grouped GPU development owner already executes single-row controls
and two-row `GroupedBroker` work on one resident model. The controls do not go
through the separate serial `Broker`. That broker owns its worker's close, and
the existing CPU observation hook is rejected for the session-bound grouped
owner. The separately reviewed owner extension implements a bounded
serial-control queue, source-stamped events and retained outputs. The
comparison will include the different serial-control and grouped-broker queue,
admission and validation costs; it will not isolate batching's kernel cost.
The existing supervisor and physical-device ownership path should be reused;
this plan does not call for a new general supervisor.

## Work and arrival model

The candidate workload uses the exact four-token input/mask schema, pinned
DistilGPT2 revision, float32/eager/eval/cache-disabled settings and seed from
`experiments/grouped_gpu_dev_v3.json`. Use the eight already disclosed synthetic
rows 103 through 110 from that file, one per logical request, retaining the
all-ones attention mask. Assign alternating rows to `team-a` and `team-b`, with
four jobs per team, and use the same request identities, rows, order, seed and
output/hash policy in both windows. These reused rows are not a new held-out
test. Preserve the relevant exposure record with the frozen run manifest.

Both modes use two concurrent clients. Each client starts after the same kind
of ready barrier and submits its next fixed request only after the preceding
response is handled. This is closed-loop offered load: the two modes need not
have identical actual arrival times, and the experiment cannot estimate
performance at a fixed external arrival rate. Record each actual attempt,
acceptance, selection, terminal and client-return time, including failed or
unattempted jobs. The grouped policy admits only the two-row combination
certified by the owner, with its coalescing deadline frozen before collection;
a selected or submitted pair is not proof that a two-row forward occurred.
Report actual call sizes, membership, wait and service intervals, counts and
successful jobs and input positions per full wall window. A single short pair
provides descriptive observations only, not a general capacity or fairness
estimate. These are processed input positions, not autoregressively generated
tokens per second.

The existing grouped owner's `_cuda` guard has process-global custody slots
and assumes a single device-driver thread; grouped broker submission itself
performs live runtime checks using that guard. A prospective driver must route
both clients through the same bounded CPU ingress/response relay, and the one
owner thread must perform all broker submissions, model calls and device
queries. Record client attempt, relay acceptance/dequeue, broker acceptance,
actual model entry/finish and client return separately. The original v1 relay
polled the broker before accepting another queued request. That schedule is
retained with the incomplete v1 attempt and must not be pooled with the
prospective `peer-relay-ready-by-head-fallback-v2` schedule below.

For v2, after the first request reaches the broker, the owner waits for FIFO
relay ingress only until the head's recorded broker-ready time plus the fixed
0.1-second fallback. A peer already queued at the decision is eligible when
its relay-ready nanoseconds are less than or equal to that deadline. The
deadline is computed as the exact rational sum of the two recorded binary64
second values, multiplied by one billion; both source clocks use the recorded
`perf_counter` implementation. The equality rule applies to an item already
queued when checked; it is not an atomic promise about a concurrent arrival.
If no timely peer is queued, the singleton fallback returns
`no_compatible_cohort` without a model forward and the pair remains incomplete.
A late peer stays in the relay.

The bounded wait uses short condition waits and stops on cancellation or the
overall window deadline. Source stamps retain relay-ready, relay-selection,
broker-ready and claim-return times. A peer that entered the relay before the
deadline may become broker-ready afterward if guarded submission is slow; the
recorded late-peer count exposes this. Scheduling and guarded calls can overrun
the nominal interval, so the fallback is dispatcher-observed, not a hard
real-time deadline. The different relay handling and its delay are part of the
measured v2 path; they do not establish a benefit from the incomplete v1 run.

Warm each actual mode and shape outside the measured windows. Freeze mode
order, warmup, all timeout and queue limits, source and asset hashes, runtime,
device identity, observation policy, row-count-specific profiles, output
tolerances, and host commitments before running. Record the monotonic clock
implementation and resolution. Begin each measured window at barrier release;
finish only after all planned attempts are accounted for and their delivery,
output offload and symmetric hash work have finished. Report queue intervals
with their precise source boundaries: the existing serial and grouped
readiness stamps are not at identical queue operations. Report synchronized
forward/service intervals and end-to-end response separately. Include timing
instrumentation and any telemetry polling overhead in the interpretation.
For the existing resident observation hook, forward entry is stamped after
input staging and before the Python model call; its synchronized finish follows
output shape checks, full-logit CPU copying, RNG cleanup and device
synchronization. That interval can include previously queued device work and
is not a GPU-kernel-only duration. The prospective owner must actually bind
these hooks or clearly report a different recorded boundary.
Use the same ordering in both modes. The primary end stamp must follow the
completion synchronization required for every delivered result and confirmed
drain. If a separate final owner synchronization or endpoint memory snapshot
occurs after that stamp, save its interval explicitly and do not describe the
earlier client-return rate as including that later operation. Aggregate and
per-team completed counts use the same full wall-window denominator.

## Correctness and evidence

For each successful actual request, retain bounded, detached CPU logits with
the job/team identity and fingerprint. Match each grouped row to its own
serial output using `abs(grouped - serial) <= 1e-6 + 1e-5 * abs(serial)` for
every element. Also verify expected shape `[1, 4, 50257]`, dtype and finiteness,
and require distinguishable serial references for a useful row-swap control.
Compare fingerprints as byte-integrity checks, not as a demand that grouped
and serial tensors have identical bits. Record actual model-entry shapes,
row offsets and call membership; at least one successful synchronized two-row
call containing distinct clients and inputs is needed for a batching claim.
Retain the numerical evidence required for independent replay. Missing,
nonfinite, misrouted or out-of-tolerance results invalidate the pair rather
than being omitted from its denominator.

The owner must independently bind its model, checkpoint, runtime, source and
physical device across windows. A common string in an event record does not
prove shared resident-model continuity. Profile and host-custody commitments
must cover the exact new instrumented source and both row counts; the earlier
grouped pilot and serial two-token profile do not authorize this workload.
Retaining sixteen full float32 rows alone requires 12,865,792 bytes; the final
host and disk bounds must also account for input copies, manifests, filesystem
metadata, temporary writes, validation scratch and any other retained objects.

## Device observation and failure

With the existing owner/session/lease guard active, synchronize before each
window, reset PyTorch allocator peaks, and record a baseline device-wide
free/total point and process-local allocated/reserved counters. After the
clients and actual worker calls have drained, synchronize again and record the
endpoint and allocator peaks for that reset window. Check each device call and
preserve partial records on failure. Process-local allocated/reserved peaks are
not device-wide VRAM peaks; free/total values are point observations, not a
continuous availability bound. Record whether external GPU/host activity was
measured and any known contamination. An exclusive cooperative slot alone does
not establish an idle device or host. Retain absolute peak counters and each
baseline: the warmed allocator can carry state from the first window into the
second, so a baseline delta alone is not an isolated batch-size memory cost.
Each allocator peak covers the interval from its reset through the endpoint
counter collection, which is wider than the barrier-to-window-end interval.

`looking_glass.grouped_throughput_memory.WindowMemory` is a prospective helper
for the existing grouped owner's single device thread. It uses that owner's
guarded calls and counter checks to write one reset baseline and synchronized
endpoint per mode, retaining incomplete counter prefixes when possible. The
owner must separately source and bind the window timestamps, prove work is
drained, inventory this new module in its source manifest and fail-stop after
an error, including an interruption before the helper's first instruction.
Its CPU fake tests do not supply a GPU profile or authorize a run.

Retain every failed, rejected, cancelled, unresolved or unattempted request and
the partial event/memory evidence. Do not silently retry, change tolerances or
start a later window while ownership or work is uncertain. A failed or
incomplete window has no valid paired speedup. The existing supervisor's
quarantine and cleanup rules remain authoritative; a successful report does
not by itself prove native CUDA teardown or future safe capacity.

## Current executable rehearsal

`python -m tools.batched_shared_cpu_rehearsal` runs the two-client job sequence
against the real serial and grouped CPU broker/event interfaces with a tiny
fabricated row model, then calls the existing `assemble_window` and
`analyze_pair`. It saves raw events in its JSON output and rejects deliberately
corrupted or row-swapped grouped output in regression tests. It requires each
two-row group's delivered outputs to fail the other member's serial reference;
a constant-output run is incomplete even if the generic pair-consistency check
passes. A fabricated serial failure retains the observed prefix and prevents
the grouped window from starting. The toy report does not persist its output
tensors, so its numeric checks are not independently replayable from JSON alone.
Its client-return stamp follows output cloning and hashing, and, for grouped
work, confirmation and disposal of the handle. Those operations are inside the
reported end-to-end window.

This rehearsal uses **two separate toy model instances**, synthetic model and
owner labels and no physical-device telemetry. Its window includes CPU broker
shutdown and dispatcher join. Its rates are fixture artifacts, not evidence
about GPU throughput, resident-model continuity or live admission. The future
GPU owner must independently establish those properties and use a timing
boundary that excludes broker teardown while retaining actual drain evidence.
The separate `tests/test_shared_grouped_same_owner.py` runs the actual CPU
`GroupedWorkspaceWorker` single-row controls and `GroupedBroker` two-row calls
with the same toy resident model and two concurrent, closed-loop clients. It
checks row pairing and retained commitments, but records no performance or GPU
telemetry.

`python -m tools.batched_same_owner_cpu` also runs this same-resident CPU relay
as an executable rehearsal. Both logical clients hand work to one owner thread.
It reports relay wait, client response and path-specific service intervals;
the serial service interval includes control execution, clone and confirmation,
while the grouped dispatch interval excludes subsequent ticket handling. It
records the toy model's observed input shapes and broker-selected group members,
including a partial prefix on a dispatch failure, but does not yet record
source-stamped broker readiness or actual model-entry times. An uncertain broker
close or an incomplete routing comparison makes the rehearsal incomplete.
Its CPU-only failure roots retain the observed in-memory prefix where disposal
has not begun; a later cleanup failure explicitly reports that disposal started.
They are not a GPU custody mechanism, and the outputs are not persisted. The
future GPU owner must save immutable output and event evidence before releasing
their claims. Toy rates, retained-commitment checks and in-memory comparisons do
not establish GPU performance, admission, or independently replayable evidence.

`looking_glass.grouped_throughput_stamps.SourceStamps` is an optional,
single-owner-thread source observer for a future instrumented owner. It uses
the existing grouped pilot's Python profiling approach and does not enable the
grouped recording scope. It records the broker's actual `ready_at` float in
hexadecimal, the `_claim` return event after internal selection with actual
member/row identities, model entry and Python return events, and the resident
`_finish_forward(True)` call. The ready clock reading and the nanosecond stamps
are separate readings; the claim-return stamp is not the exact queue-pop time.
The resident's pinned CUDA source calls that finish hook after its guarded
synchronization, whereas the CPU fixture has no CUDA synchronization. A Python
return event can occur while unwinding an exception, and the finish hook does
not establish a successful result. The live owner must reconcile expected
counts, source, clock, outputs and completion state, reject a lost profiler and
retain partial evidence on failure. Profiling overhead occurs throughout the
instrumented owner-thread interval and is part of both measured paths; the
helper does not establish GPU execution, cleanup, admission or isolation.
Successful stamp records contain only scalar values, but an exception and its
traceback can retain a frame or tensor. The live owner must keep that failure
graph, its device/host custody and its lease rooted until supervised recovery.

`looking_glass.grouped_throughput_driver.run_pair` composes a single-owner
relay, the source observer, a supplied guarded call, two memory windows and a
supplied private-row writer. It checks the exact owner/resident/authority and
two lease objects, request and seed digests, paired source events and delivered
identities, and every matched and cross-reference comparison. Each wall window
ends after clients drain and the supplied final synchronization succeeds.
Full rows are written after that wall end and before the endpoint memory point;
allocator peaks therefore include the intervening CPU persistence interval.
The report records the wall-end-to-snapshot start and finish delays; endpoint
free memory is a later point sample, not an instantaneous window-end reading.
The serializer receives independent copies. A failed endpoint retains the
already published row records; failures before persistence may have only a
scalar prefix and rooted in-process objects. The caller must pre-reserve the
host and disk envelope, root the owner/custody/exception and fail-stop; the
driver itself neither obtains a device lease nor supervises, launches or
attests the process. Its CPU tests do not approve a GPU source or profile.

With all sixteen results present, the relay keeps both the owner's output copy
and the client copy: their data alone can total 25,731,584 bytes. The actual
host envelope must also cover overlapping worker/ticket results, inputs,
finite/comparison scratch, metadata and runtime overhead; an accounting claim
is not an operating-system memory cap. The private row payload on disk is
12,865,792 bytes before metadata and filesystem overhead.

## Disabled fixed-owner integration

The prospective extension is disabled at both the grouped owner and extension
entry. A separately source-bound controller must protect the complete owner,
bootstrap and throughput source tuple before acquiring the device or importing
the model. It must use a fresh admission and preserve the existing physical
device lease, strict supervisor, direct-owner exit and cleanup requirements.
No result is approved merely because this CPU-tested extension exists.

After the existing two sampled and two held-out validation groups have passed,
the extension uses the same resident model, worker, authority, broker and
device commitments for both windows. Each of two clients submits four jobs in
closed loop, one outstanding job at a time. Each row has four input positions,
an all-one attention mask, the fixed seed 2026102808, and the already exposed
rows 103–110 from the grouped development protocol. In order by client round,
the pairs are `[50256, 103, 203, 303]` / `[50256, 104, 204, 304]`, then
105 / 106, 107 / 108, and 109 / 110 with the corresponding 200- and 300-series
positions. The first window uses eight single-row owner controls; the second
uses four two-row broker groups. A missing group, indistinguishable cross-row
reference, wrong delivery, changed input or failed comparison makes the pair
incomplete. Both windows include their final guarded synchronization. Each has
a 30-second dispatcher-loop deadline, checked between operations, and 15-second
bounded client/barrier waits. A timeout leaves an incomplete observation; the
loop deadline cannot interrupt an operation already running. Successful windows
do not wait for those deadlines. Rates use the complete barrier-to-drain wall
interval. These are requests/s and input positions/s, not generated tokens/s.
Each window record and the paired result retain the local `perf_counter` and
`monotonic` clock implementations, resolution and adjustment properties.
Profiling, relay, validation, offload, polling and filesystem or other host
contention affect the measured paths; the serial control bypasses the serial
broker.

The extension opens a separate 64 MiB cooperative host claim before building
the fixed inputs and retains it until all sixteen row files are published,
comparisons are bound to the saved hashes, and both retained CPU output copies
per job have been released. The grouped owner's existing control and broker
ledgers remain separate. These claims do not limit RSS or reserve physical
memory; the controller must check fresh host headroom. The owner bounds its
memory evidence to 2,048 records of at most 4,096 serialized bytes each. The
row writer bounds the sixteen raw rows and their metadata, and window/pair
records are limited to 256 KiB each. A fresh artifact-volume reserve of at
least 128 MiB covers these bounded contents plus filesystem overhead; any
checkpoint staging, cache, bytecode or other-volume writes need separate fresh
space checks. An incomplete endpoint preserves already written rows and roots
the known owner, lease, claims and exception; earlier failures may retain only
an in-process prefix. No point sample is a device-wide peak or native cleanup
attestation, and warm allocator state can carry from the first window to the
second. A separate saved-file replay must rehash and compare the retained rows.
