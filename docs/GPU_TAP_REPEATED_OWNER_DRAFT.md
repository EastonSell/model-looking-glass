# Repeated tap owner admission draft

This is a CPU-tested component for the proposed
[repeated tap study](GPU_TAP_LATENCY_REPEATED_V1.md), not a GPU owner or an
authorization to run it. The study's CUDA entry remains disabled. The earlier
contaminated benchmark remains separate.

`tap_repeated_admission._Admission` is a private, single-threaded, single-use
custody object. A future supervised owner must construct and strongly root it
before calling `acquire()`, supply actual session/device-lease checks and live
telemetry, and stop the process on any escaping failure or interruption. The
component itself neither authenticates the callback nor initializes a device.

It validates each sampled free/total/allocated/reserved record against the
fixed reserve and remaining-cap guard and requires at least 3 GiB free before
model transfer. It roots a `MemoryBudget` before acquisition, commits the
whole 25% allocator cap plus 256 MiB headroom plus a separate 256 MiB
outside-cap allowance, and reserves 128 KiB of aggregate GPU trace before
model, input, event or trace allocation. `MemoryBudget` also applies its own
reserve check, so this admission can be more conservative than the protocol's
sampled guard. A lost acquisition return leaves the issued lease in the rooted
budget. A lost trace-reservation return leaves the budget and any committed
trace claim rooted. These are logical commitments and point samples, not a
physical reservation, a bound on CUDA intermediates, or a host-memory cap.
The future single owner must separately bound and account for its CPU
checkpoint/model before transfer, host input, results, trace metadata, journal,
telemetry and subprocess buffering. The 128 KiB claim is for source-device
trace values only. Until an exact host envelope and physical-availability
checks are measured and enforced, peak process RSS remains unbounded by this
component and is not an admission guarantee.

The object's `verify()` checks the exact active budget, lease and trace claim;
`check_snapshot()` samples again with owner checks before and after. The
private `release_after_disposal()` assumes its trusted caller has already held
all outputs and captures through successful synchronization, disposed the
model/input/events/results/traces, and successfully synchronized that disposal.
It cannot observe or prove that precondition. It releases the trace claim and
then the whole-run lease while continuing to check owner custody. Verification
and telemetry use nonactive transition markers while their checks run; the
tested interruption in the verification failure handler leaves that marker
nonactive. The marker is cleared before successful return and before a later
verification, so multiple interruptions can still leave a healthy-looking
state. An exception or interruption requires process termination, retaining
known budget/lease references; a release interrupted after its commit is not
retried.
If the whole-lease release committed before its return was lost, the ledger
lease may already be gone. The physical device and supervisor custody must
still be retained, and that failure cannot produce a clean attestation.
The physical `DeviceLease` must remain held through owner exit, and supervisor
control must remain held until reap.

The caller must use the actual `OwnerSession` and canonical `DeviceLease`,
check their exact identities and runtime UUID, retain model, input, events,
trace sessions, outputs and failure graphs, and stop all CUDA calls on lost
ownership or uncertain synchronization. Any interruption after a transition
commits but before its return is still ambiguous to the caller: the owner must
exit rather than infer that a healthy-looking state is safe. Python cannot
guarantee rooting a native value interrupted before assignment or an exception
interrupted before it is rooted. A strict zero-exit/reap supervisor and an
external watchdog remain necessary. The CPU-only fake-device guard in the
scaffold must not be used as this owner's safety mechanism.

`tap_repeated_owner._OwnerOperations` is a separate, disabled operation seam.
It requires exact common session/lease/admission classes, checks the physical
identity and active commitments on both sides of an operation, and can root an
assigned result in a designated slot before the second check. Its preallocated
slots accommodate four preflight result objects, three trace sessions and two
captures concurrently. If a call fails after Python has assigned a return but
before its designated slot is filled, its failure handler attempts to retain
the value in a separate emergency slot independent of the exception traceback.
When that handler completes, it roots the original exception privately and
raises a fresh public exception with no implicit context. Clearing the public
traceback therefore does not clear the private operation's nested frames.
This is a best-effort single-fault boundary, not a guarantee against an
interruption before the private root is stored or against frames already
discarded by another component.
A further interruption before that emergency assignment remains uncertain. Its
clearing method checks custody before and after each root drop and stops on a
detected loss; these are point checks and loss can race an individual drop. It
assumes completion synchronization and does not itself establish that disposal
completed. Tests replace the common
classes with CPU fakes; they do not exercise CUDA or authenticate a supervisor.
The future top-level owner must root this object before initialization and exit
after any escaping fault, even if an interruption leaves a healthy-looking
internal state. The class is not a runnable benchmark or a general recovery
mechanism; native return-to-assignment gaps remain.

The disabled seam can pre-root a private `_RetainedEvents` container for each
fresh bounded `GPT2TraceSession`, before entering it. The trace may append at
most two actual events. Its normal failure-time `events.clear()` does not drop
the privately retained event objects or their captured tensors; a failed trace
is still unsuccessful and must never be published as a valid result. The owner
checks that its three preflight slots contain, respectively, ID-only, direct,
and gradient traces, each selecting exactly block 0, sample 0 and token 1,
with the fixed v1.2 session ID and 64 KiB trace limit. Thus at most two
preflight sessions retain values under the 128 KiB claim. A measured trace
cannot be entered while any preflight trace/container remains, and is limited
to one direct or gradient session at a time. These checks apply to the
pre-rooted session; the future owner must use this seam before entering it.
The owner
must retain both session and container through completion synchronization, then
drop their aliases only on the synchronized disposal path. Its final teardown
rejects any still-occupied trace/container roots. The actual timed path must
include and report the direct/gradient container overhead rather than treating
the container as free. This protects values already appended to the container.
Selection and gradient temporaries created before append may be retained by
the original private exception graph when its Python frames are available,
but expression/native return-to-assignment and interrupted error-rooting gaps
are not covered. In those cases the owner must stop and retain known claims
until hard process exit; there is no complete transient-custody claim.
Importing `tap_repeated_owner` alone does not import Torch; the trace types are
imported inside an owner-checked operation. The future bootstrap must also
ensure it constructs the trace only after physical ownership is bound. A
guard around a model/trace call checks the owner at its top-level boundaries;
it cannot interrupt every individual PyTorch operation or atomically prevent
mid-forward ownership loss. Such a loss leaves completion uncertain and
requires process replacement, not a claim of immediate device cancellation.

The disabled `tap_repeated_gpu_owner.owner_main` draft connects these seams
to the exact supervisor, source bootstrap, frozen schedule, parent barriers,
86-row durable journal and fixed model. It checks the shared workspace and
lease before importing Torch, and keeps the lease and known resources rooted
through the bootstrap's direct hard exit. It disables automatic cyclic GC
before importing Torch and forbids explicit collection after that import, prior
to creating CUDA resources. A still-live weak reference blocks healthy
release. These rules do not measure or cap host RSS. Journal records are
bounded; the resident checkpoint and dependency memory have no separate host
envelope. CPU-only fakes exercise the orchestration but do not attest CUDA,
the actual process lifecycle, or the absence of hidden native allocations.

The candidate records the claims still held after the last sample and the
actual model, runtime, source and journal identity. The final record follows
parent cleanup authorization, synchronized disposal and claim release; a
fresh lease check immediately precedes its write and clean attestation. The
lease-through-exit statement depends on the reviewed rooted-owner/direct-exit
control path and parent reap, not on a post-exit lease observation. Failure
records include a bounded stage and exception type; raw failed telemetry and
unretained native temporaries may be unavailable. After any escaped error the
child takes no further CUDA action and does not attest or retry cleanup.

Before enabling a real study, complete and independently review the combined
source/runtime/checkpoint and live process-handle binding, guarded CUDA
execution and synchronized teardown, the frozen timed path, parent-owned
activity collection and gaps, failure journals, parent replay, privacy
projection, and the whole exact-path memory admission. A separately announced
isolated timing window and GPU recovery are required. No output from this draft
is a timing or GPU performance result.
