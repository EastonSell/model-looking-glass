# CPU-only scaffold for the repeated tap benchmark

The proposed experiment is specified in
[GPU_TAP_LATENCY_REPEATED_V1.md](GPU_TAP_LATENCY_REPEATED_V1.md). This
scaffold exercises its saved order, journal checks, paired statistics and
sampling boundaries. It is **not the supervised GPU runner**. `run_gpu()`
raises before querying or initializing CUDA. No pretrained model is loaded.
The original contaminated benchmark remains separate and is not pooled.

`tap_repeated_protocol.load_schedule()` checks the SHA-256 of the exact saved
schedule bytes. Before each use, `session_order()` checks a canonical snapshot
again so a mutated in-memory schedule is not silently accepted. Each of the
five session slots has two warmup rounds and 12 measured rounds, with each mode
occurring once per round and four times in each position in the measured set.

`tap_repeated_runner` has an explicit CPU simulation using synthetic tensors,
synthetic tap attribution and fake device memory/event counters. It uses the
real CPU `gradient_readout` operation, but does not attach GPT-2 hooks or
measure CUDA. Its output is labeled `synthetic_cpu_simulation` and
`gpu_eligible: false`; its event and memory numbers must never be promoted to
GPU evidence. For example, using a fresh directory:

```text
python -m looking_glass.tap_repeated_runner --schedule evidence/gpu-tap-latency-repeated-v1/schedule.json --session 0 --output <new-directory>
```

Separately, `tap_repeated_cpu_trace.cpu_trace_preflight` exercises the real
`GPT2TraceSession` ID/direct/gradient hooks on a caller-supplied eager CPU
GPT-2 model and the fixed token IDs. It checks output and capture equality
using `torch.equal`, event identity, and shape/byte metadata. The focused test
uses a small randomly initialized model, so it neither verifies the pinned
checkpoint nor meets the protocol's hidden/output layout. The test also runs
one sample per mode through the same timing/disposal core with those real CPU
hooks and fake device events/counters. Its small-model rows are intentionally
incompatible with the fixed GPU evidence contract and do not authorize CUDA.

One simulated owner is allowed per process. The fake guard checks the owner
around each operation and marks it in flight until shortly before return. If
an operation fails, the marker remains visible when the first failure-handler
assignment is interrupted. This is only a local best-effort check: if the
return is interrupted after the marker clears and a second fault interrupts
the handler, the guard may appear healthy and a later call can overwrite its
return root. Interruptions before marking an operation or rooting an exception
have similar gaps. The outer driver must stop and replace the process after
any escaping interruption, even when the guard appears healthy; it must never
catch an interruption and continue. The fake guard is not a GPU custody
primitive or a substitute for an external watchdog. Preflight retains all four
fake outputs and the ID/direct/gradient
sessions and two values through completion synchronization, disposes them and
synchronizes again. Each sample similarly starts after a synchronization,
records a stream event, performs and validates the operation, records the end
event, synchronizes with output/capture alive, disposes them and synchronizes
again. Memory checks and fsynced pending-observation and separate postcheck
records are outside the wall interval. A failed postcheck leaves its pending
record and cannot form a complete journal. `simulation.json` alone is an
uncommitted candidate: `inspect_cpu_simulation()` requires the matching
`completed.json` hashes, complete journal, an externally observed zero exit and
reap, and no conflicting `failure.json`. Conflicting terminals are incomplete.
The caller's exit/reap observation is trusted rather than authenticated. As
with any Python wrapper, an interrupt between a native return and assignment
can prevent rooting the returned value; the fake-device checks do not prove GPU
failure safety.

`tap_repeated_protocol` can recompute per-round ratios, differences and
per-mode wall/event medians from a complete ordered journal, after checking
the clock, output/trace metadata and sampled memory guard. It rejects a device
event span exceeding its enclosing wall interval by more than 100 microseconds;
this allowance is a conservative consistency tolerance, not a measurement of
clock accuracy. It also requires the producer's finite-output and, for traced
samples, finite-capture assertions; those assertions do not independently
prove the contents of tensors that were not retained. Its activity-screen
helpers distinguish incomplete availability from contamination and expose a
PID-free projection. The five-slot aggregation requires distinct attempt slots,
matching child/parent supervisor run ID, PID and Windows process creation time,
and a parent assertion that it observed a live process handle. Every slot
claiming a reused run ID or reused PID/creation-time pair is incomplete, even
if a conflicting attempt has mismatched identity records. Reuse of a PID with
a different observed creation time is allowed. The activity screen's own PID
must match the owner. It also requires recorded synchronized disposal, the
device lease held through process exit, supervisor control held through reap,
matching externally supplied binding, a complete screen and all postchecks.
It reports no aggregate with fewer than three complete sessions and uses the
median of session medians. These are
consistency checks on trusted supplied records: they do not authenticate the
source, model, device, processes or execution. Fabricated records can pass;
the future real parent must independently obtain and bind the live process
identity and strict supervisor result rather than assert these fields.
The activity consistency check requires the post-workload process query to
finish within five seconds of workload end and limits its duration to two
seconds. This fixed screen is a freshness bound, not proof that no other GPU
activity occurred between observations.
The memory check rejects a snapshot whose reported free plus reserved bytes
exceed the reported total. Real telemetry reads would be sequential rather than
an atomic physical-memory snapshot, so such an inconsistent read is rejected
conservatively.

Before any GPU execution, a separate implementation must connect the real
fixed DistilGPT2 workload and `GPT2TraceSession` to a reviewed supervised
owner, independently bound physical-device lease and runtime/source identity,
the provisional allocator/trace/host commitments, fresh admission guards,
real CUDA synchronization and event APIs, and actual activity collection. It
must preserve roots and claims through verified disposal, stop all CUDA calls
after lost ownership or uncertain synchronization, and retain the lease until
the owner is reaped. It also needs parent replay, bounded failure journals,
privacy projection and an independently verified exact resource profile.
The existing untraced two-token service profile does not authorize this path.
