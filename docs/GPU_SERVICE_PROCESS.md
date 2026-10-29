# Fixed supervised service child

`looking_glass.gpu_service_process` is a fixed two-token owner intended to be
invoked only as a direct child of the identity-aware supervisor. It is not a
general-purpose GPU service launcher. It cannot run the agreement study, and
the separately measured worker profiles and reviewed controller composition
are not yet available. Do not treat the direct-model GPU pilot as a profile for
this owner.

The child first reads and checks the exact private config bytes pinned by the
parent. Before acquiring ownership it rejects previously imported Torch,
Transformers or Hugging Face Hub modules, including submodules: Hub caches
its home at import time, so a later environment change cannot correct an
earlier import. The trusted launcher establishes the cache environment in a
fresh process; this check does not authenticate the interpreter or libraries.
The child then obtains the actual `DeviceLease`, checks its `OwnerSession` and
workspace identity, and reconstructs trusted source, checkpoint and runtime
identity. Three independently approved load, cold-forward and warm-forward
profiles must match that live identity, the fixed input `[15496, 995]`, and
their protocol-specific seeds and case hashes before model loading. The owner
verifies the loaded CPU model, synchronizes the copy and verifies the tied GPU
parameters before constructing its worker and bounded IPC endpoint. Each
client request must match the fixed input and warm seeds; arbitrary prompts,
batching, tracing and interventions are outside this pilot.

Readiness credentials are delivered only through the supervisor's private
startup pipe. The owner retains the model, device lease and resident
reservation until direct process exit. It attests synchronized quiescence only
after the worker and broker drain; it does not claim that allocator counters
are zero while the resident model remains alive. The supervisor reaps the
exact child and reacquires the lease before clearing a valid clean marker. Live
headroom and the reserve must still be checked for every future admission.

The child exits directly with `os._exit` on both normal and exceptional paths,
so Python finalizers and `atexit` callbacks do not run while device ownership
is being relinquished. Its post-owner exception handler does not write to a
potentially blocked stderr pipe; argument parsing before acquisition and native
dependencies can still produce output. Bounded exit codes distinguish invalid
configuration (20), worker failure (31), quarantine or an owner exception
without a verified result
(32), incomplete shutdown (33), stop-monitor failure (34), an unfinished
monitor (35), a reported memory failure (36), and an uncaught base exception
(70). An unverified drain takes precedence over a reported worker failure or
memory error. A memory-specific code is emitted only when that category is explicitly
available; a worker OOM may instead appear as generic worker failure (31) or
quarantine (32). A status code is a workload classification, not proof of
teardown or a complete error taxonomy; consult the supervisor result and sticky
marker.

CPU tests cover config pinning and rejection before owner entry, preloaded
dependencies, status classification and direct exit without `atexit`. They do
not initialize CUDA, authenticate native code, verify a real profile, or
establish that the complete GPU service is runnable.
