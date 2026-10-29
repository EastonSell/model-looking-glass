# Supervised GPU service composition (in development)

The owner composition in `looking_glass/gpu_service.py` is **not a public
launcher or a measured GPU service**. Its supported target is the pinned
DistilGPT2 checkpoint and one exact no-capture forward signature: one batch,
the `int64` input IDs `[15496, 995]`, a 512 KiB tensor output cap, and a 4 MiB JSON response
cap. One seed-17 cold forward on `[15496, 995]` is intended to precede warm
requests. The initial service accepts only the same input and warm seeds 29,
43, or 71; broader requests need matching evidence or a reviewed bound. The
worker serializes model execution; multiple clients may submit
work, but this is not evidence of concurrent GPU kernels, throughput, or a
safe upper bound on memory use. Trace retention is zero in this initial mode.

The staged owner checks the supervised device lease and the runtime UUID before
telemetry, resolves separate load/cold/warm profiles against independently
constructed live keys, retains the model-load reservation until process exit,
and repeats the lease, runtime, and headroom checks before copying weights.
The fixed two-token service holds a 2 GiB load commitment and one separate
512 MiB workspace commitment, acquired before the cold warmup and retained
through process exit. Every serial forward checks those exact live capabilities
and current headroom. The cold result is validated before the worker advances
to the separately bound warm profile.
The measured estimates must fit within those commitments; they do not replace
the fixed reservations or guarantee that future operations cannot run out of
memory. The CPU generator is seeded with 17 immediately before offline loading,
matching the measured load stage. Ownership is checked between individual CUDA
identity and telemetry calls and before and after synchronization.
Every budget telemetry snapshot must report the same 7–9 GiB device total
bound by all three live workload keys; a mismatch fails before admission.
The exported `FIXED_SERVICE_OWNER_CONTRACT` value
`fixed-two-token-commitments-v1` lets a controller reject incompatible owner
versions before startup; it does not replace source and profile verification.
The actual CPU model must pass an explicit verifier before that copy. Both
PyTorch module-conversion flags must read back as false around the copy, and
the converted model must pass an independent, trusted device verifier after
synchronization and before worker construction or warmup. The intended
verifier checks that GPT-2's embedding and language-model head are the same
device `Parameter` and share storage under the live supervised lease. The
intended verifier reads metadata and storage identity without allocating GPU
tensors, transferring data, or launching kernels; its exact implementation
must be included in the measured startup envelope. The owner synchronizes
again after a successful check. This is a required startup callback; a fake
callback in CPU tests is not GPU evidence.
These checks rely on exclusive, cooperative ownership of the process and model
and on the verified model's trusted conversion implementation; they do not
protect against arbitrary concurrent mutation. If an unusual conversion returns
a different object, the owner retains it after assignment and fails. An
asynchronous interruption between the return and that assignment can prevent
retention of that returned object; the owner must quarantine and exit. At
shutdown, the owner distinguishes workload failure from verified device drain;
worker or broker quarantine prevents attestation, including a quarantine
observed after synchronization. Unverified state latches process quarantine
and retains known owner resources. A preallocated process-lifetime custody
root holds the budget before lease admission, the model-owning backend, and
other known resources without relying on an allocation during failure handling.
One process is used for at most one owner and must exit after its result. This
does not prove retention of CUDA temporaries internal to a failed operation.
Once a drain fails, the process latch prevents retries and clean attestation.
The final recheck relies on the service having stopped every potential
signaler; it is not an atomic lock between an independently running signaler
and attestation. An uncertain shutdown must remain unverified.
The launcher remains gated while this signaler-quiescence/attestation boundary
and the fair ingress and IPC custody integrations are being completed. No
compatibility marker for that completed boundary is exported by this draft.

The fixed worker and Broker share a sticky completion authority. A success,
retirement, buffer cleanup, or guarded device-operation decision is ordered at
its unsafe-flag read; an unsafe signal first suppresses that decision and
retains its known custody. A decision first may physically finish or become
visible after a later signal. The signal cannot preempt an in-flight native
operation. CPU/GPU producer buffers are staged before disposal, and last
references are released outside the decision lock. These guarantees require
the service's unsafe sources to mark the same authority synchronously, with
one trusted resident worker in the owner process. A separate process-quarantine
latch is rechecked during model retirement, but an independent source that
does not mark this authority is ordered only when detected. An
independently revoked physical session is detected at a point-in-time lease
check, not atomically at the instant of external revocation. The flag's fast
publication relies on the supported CPython execution model, not a general
cross-process atomic or an OS memory limit.

Each bound CUDA operation checks the actual owner session and physical lease
at its decision boundary. CUDA RNG restoration is skipped after unsafe, and a
final guarded synchronization follows successful restoration. An unsafe or
failed drain keeps known buffers and both commitments through process exit;
it cannot recover unknown device-internal temporaries or a newly allocated
host tensor interrupted before Python assigns it to a custody slot. A later
snapshot failure may also unwind earlier clones from a partial dictionary on
the interpreter stack; retaining the exception does not recover those clones.
Any producer retaining an alias of such a clone must account for and retain it
independently. Such a failure requires process quarantine; it is not proof of
disposal. Host result, IPC and
consumer ownership require separate claims and disposal proofs. A fixed
workspace or bounded output size alone does not establish aggregate host or
native-device memory consumption.
Bound producer and model-state inspection exceptions are privately rooted
when observed; the caller must not infer safe disposal from a sanitized error.
The budget's telemetry callback is trusted owner code: it must independently
check that same session and lease before each CUDA query. The fixed worker also
checks the authority immediately before and after the budget snapshot; these
outer checks do not replace the callback's per-query guards.
The session guard and telemetry callback must themselves retain any
resource-bearing partials and their associated claims on failure through
process exit. Their exception tracebacks can reach callers and are not private
custody roots. The concrete callbacks must be reviewed against that contract
before the service is enabled; unknown native temporaries remain outside it.
The fixed workload's recorder policy is disabled. The bound worker rejects an
attached broker serial recorder and checks the executing thread for an active
grouped or serial forward-observation scope; a detected change marks the
owner unsafe. A future recorded workload needs its own verified identity and
profile. The collector can detach a closed, unbrokered worker from its
authority only after execution and custody have stopped. This retirement is
one-shot: it retains the authority and unsafe latch, forbids rebinding, and
does not attest process quiescence or device disposal. The collector must keep
its own worker, model and lease roots until disposal actually finishes. On any
failure or interrupted return it must mark the authority unsafe and retain
those roots through process exit, because detachment may already be committed.
These ordering checks are CPU-tested with fakes, including real supervisor
subprocesses and an authenticated two-team CPU service. They do not establish
that the GPU path works on this machine.

Real service launch still requires a reviewed CPU-only preflight, a trusted live
source/runtime/model verifier, a private supervisor-to-parent startup channel,
and separately reviewed measured model-load and forward profiles. None of the
present test profiles is measured or approved. The measured profile must cover
the same authority-bound `LifetimeWorkspaceWorker` and resident custody/RNG
path; an unbound `ResidentWorker.execute` profile is a different workload even
if source hashes and shapes match. All real owners on this machine
must use the same canonical project workspace root for device coordination.
The generic test profile loader does not prove that its evidence used the
specified token case and stage seeds; a production verifier must check both.
The runtime UUID representation also requires verification against the actual
device under supervision. A missing prerequisite must fail before launching an
owner; an uncertain owner must exit with its sticky marker intact.

The shared direct-child bootstrap in `looking_glass/gpu_bootstrap.py` suppresses site initialization and
pre-execution `PYTHONPATH`, keeps the standard library before project paths,
and rejects sourceless bytecode and origins absent from its source manifest.
It compiles project and environment Python code from the same bytes it hashes.
For native extensions it checks bytes before the OS opens the path; that is not
an atomic native-load verification. The frozen checkout, environment and
private cache directory must remain trusted and unchanged for the owner
lifetime. The source manifest also does not hash every OS or driver component.
These checks do not create a sandbox against arbitrary native code.
An opt-in CPU smoke exercises this shared guard with a real offline checkpoint
and a synthetic dry-run inventory; this is not a production source manifest,
GPU profile, or validation of CUDA-specific imports.

## Development incident: unintended CUDA query

During a failing development regression at approximately 2026-10-29 00:57–00:58
UTC, `test_model_verifier_must_explicitly_return_true` invoked the real backend
with a fake session. The implementation had not yet rejected a verifier's
`False` return, so it reached `torch.cuda.device_count()` and
`torch.cuda.get_device_properties(0)` before fake identity verification failed.
No model was copied to the GPU; no benchmark, calibration, or profile evidence
was collected. No coordinated GPU window was active.

The command was `work\.venv\Scripts\python.exe -m unittest discover -s tests
-p test_gpu_service.py -v` in `work\teams\gpu_service`. The command runner
reported exit code 1 after 7.7577 seconds (output chunk `92ae2a`), but did not
report the Python PID. Attempts to recover a matching Sysmon or Security 4688
event yielded no event, so the PID is **unknown**. A subsequent Windows process
query found no Python command line containing `gpu_service`; a read-only
`nvidia-smi` process query listed no Python process. Windows WDDM reported
per-process memory as `N/A`, so neither observation proves exclusive GPU use.

At 2026-10-29 00:59:03 UTC an independent read-only `nvidia-smi` snapshot
reported 8192 MiB total, 203 MiB reserved, 871 MiB used, 7119 MiB free, and 0%
utilization. It is a point-in-time observation, not an isolation guarantee.
All timings before a fresh quiet window remain non-benchmark data. The model
verifier now must return exactly `True`, and the CPU test module hides CUDA
before importing PyTorch, blocks PyTorch CUDA/driver entry points, CUDA tensor
copies, NVIDIA library loads through `ctypes`, and `nvidia-smi` child commands,
and installs the same call guard in fake supervised children. These test guards
are not a sandbox against arbitrary native code.
