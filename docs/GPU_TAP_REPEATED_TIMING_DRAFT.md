# Repeated tap timing core, disabled draft

`tap_repeated_timing._TimedSample` is an experimental, disabled sample core.
Its tests use a tiny real CPU GPT-2 model and a fake device/event backend; they
do not measure a pretrained model or initialize CUDA. There is no real backend,
preflight, checkpoint loader, real device memory postcheck, parent grant, journal writer,
supervised entry point or GPU launch in this component. It is not an experiment
result or an authorization to run one.

Before each mode it checks all preflight roots are empty, synchronizes the
device, resets peaks exactly once, and samples free/total/allocated/reserved
memory outside the wall interval. Each read is separately guarded and its
basic validity is checked before the next device-facing read. After wall end,
it reads peaks and a second memory snapshot, then source-stamps completion.
The actual CUDA callbacks remain unimplemented. For each mode it performs the
same number of top-level owner/admission checks:
the no-capture path substitutes two checked no-ops for the private event
container's allocation and binding. After a checked default-stream observation,
the wall interval starts immediately before a checked start-event record. It
contains trace setup, one forward, output/capture validation, a second checked
stream observation, end-event record, completion synchronization with model
result/output/session/capture/container roots still live, event elapsed read,
individual root disposal, a weakref check that supported transient objects no
longer have outside aliases, and a second synchronization. The wall interval
ends at the following source-clock reading. Device events and elapsed values
are supplied by the trusted fake backend in the current tests. A failed or
interrupted step latches unsafe, retains known roots and prevents another
sample; the outer owner must replace the process on any escaping interruption,
including an interrupted return or failure handler.
The bounded partial record retains validated pre-memory, output/capture
metadata, source timestamps, event duration and each validated peak/post-memory
value as it becomes known. It does not publish tensor contents; interruption
before a field is assigned or before a private failure root is stored remains
unverified. A future hard-exit owner must persist whatever partial record was
actually completed without fabricating missing fields.
Matched check counts do not remove their cost or host scheduling variance;
for two-token workloads they can dominate or dilute the measured difference.
CUDA-event time is secondary and covers a different interval. Neither value
would, by itself, isolate a hook-only cost.

The component binds one schedule-session index and requires automatic cyclic
collection to have been disabled and latched before sampling; every guarded
operation checks it remains disabled. This does not prohibit a dependency from
calling `gc.collect()` explicitly. Top-level guards cannot prevent a marker
change in the middle of a model call or atomically guard every PyTorch/native
operation; the trusted owner must retain claims and stop when that uncertainty
is detected. Native return-to-assignment and repeated-interruption boundaries
remain outside the demonstrated Python root guarantee.

Actual GPU integration still needs exact source/runtime/physical UUID binding,
the measured fixed checkpoint and input, stream/event calls, per-boundary
memory admission and postchecks, a fixed preflight that enters the three
approved trace slots, a reviewed loader handoff, source-stamped evidence,
parent-owned telemetry, a no-device-work screen barrier, synchronized
disposal and direct hard exit. The actual loader/import order and all owner
aliases must be checked with cyclic GC disabled; tiny CPU tests do not prove
that GPU events, native workspace or the pinned loader will become collectible.
The native CUDA allocator and process RSS limits also remain unmeasured for
this instrumented workload.
