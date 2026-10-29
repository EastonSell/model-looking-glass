# Repeated tap CUDA backend, disabled draft

`tap_repeated_cuda_backend._CudaRuntime` is a private, disabled CUDA-call
adapter for the prospective timing core. Importing it does not import Torch
or initialize CUDA. Its tests replace all CUDA calls with CPU fakes; no real
device, event or memory reading has been validated. This is not an owner or
authorization to run the study.

Only after an exact supervised owner, physical lease, runtime UUID, active
admission, schedule slot and disabled cyclic-GC policy have been established
may a future integration consider initializing the adapter. Its guarded
initialization sets and reads back the 25% PyTorch allocator fraction, records
the device-0 default-stream object and identity, and pre-roots two distinct
timing events. Both event records explicitly receive that same stream object.
It rejects a stream or events that do not support weak references before any
preflight; synchronized teardown includes the stream root. It records both
events once and synchronizes while the owner is guarded before preflight, so
their lazy native initialization is outside timed warmup intervals.
The timing core is expected to invoke every backend method through the exact
owner guard; direct calls to this private adapter do not independently provide
that guard. Its methods bind to the original owner/session/lease/admission,
the original event roots and one calling thread. Stream checks and event
records are point observations, not atomic prevention of mid-call revocation.
The two current-stream wrappers observed in each sample have separate owner
roots through the completion synchronization. They are dropped and checked
alongside the other sample resources before the post-disposal synchronization.
These extra guards and wrapper checks occur in every mode and are part of the
measured instrumented path; equal counts do not establish equal overhead.

Each memory read, peak read, reset and synchronization is a separate
CUDA-facing callback. The timing core validates each returned value before
issuing a later dependent query and holds known resources on failure. The
adapter can only root assigned returns and exceptions on the documented
single-fault boundaries; an interrupted native return before Python assignment
or a second interruption in a failure handler remains uncertain. Its
initialization gate and the timing gate stay disabled until the full pinned
owner, source/runtime binding, parent barriers, journal, synchronized disposal,
strict zero-exit supervisor and physical lease lifetime are reviewed together.
