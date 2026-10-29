# Resident model execution contract

This is a deliberately narrow building block for a shared-model service. A
`ResidentWorker` owns one preconstructed, trusted PyTorch model, and runs one
read-only forward at a time. It does not provide a server, scheduler, admission
control, cross-request batching, tracing, interventions, or parallel CUDA
execution. A caller-supplied model label is not proof of checkpoint identity;
the model pool must verify the full identity before reusing an instance.

The owner must prepare the model on the intended device, put every module in
evaluation mode, and give the worker exclusive access to it. Model code and
construction are trusted configuration, not part of a submitted request. A
request supplies only a nonempty ID, a seed, and named CPU tensor inputs
(`input_ids`, plus optional `attention_mask`, `token_type_ids`, and
`position_ids`). Inputs have the same nonempty two-dimensional batch/sequence
shape and are subject to configured batch, sequence, and byte limits. No
request-supplied function, hook, model, intervention, or generation cache is
accepted. A single request may include a bounded batch; different clients are
not combined. The public model label, device, and configured bounds are
read-only properties; trusted code can still mutate Python internals, so this
is not a security boundary.

The broker validates and snapshots inputs before enqueueing. Callers must not
mutate tensors while the snapshot is being made. The worker checks again and
copies its inputs, runs with gradients disabled, and returns a separate,
detached CPU logits tensor in contiguous storage with a zero offset and exactly
its logical byte size. The output cap applies to that tensor, not to
model weights, temporary activations, non-logit model outputs, Python metadata,
allocator overhead, the source logits and other transient copies during the
forward, or memory retained by completed results. In particular,
per-request limits alone do not bound results retained across many requests;
the broker or consumer needs its own retention and backpressure policy.

Requests use seeded PyTorch CPU and target-device RNG state, restored after the
forward. A process-wide worker lock protects this state against other workers
using this component; unrelated threads that touch the same RNG concurrently
can still race. Use a dedicated execution process, and do not make claims of
bitwise reproducibility across devices, library versions, backends, or
nondeterministic kernels. The model must be a read-only inference model. The
worker checks for registered module hooks and common registered state changes,
but this is only a best-effort guard: for example, mutation through `.data`
can bypass tensor version counters. It cannot sandbox arbitrary model code or
prove checkpoint identity, immutability, absence of external references,
unregistered mutable state, or unsupported side effects. The owner must also
prevent concurrent installation of process-global hooks. The public
`registered_model_state(model, device)` helper exposes the same best-effort
checks for an owner holding a model lease; its identity/version snapshot is
only comparable within the same process and is not an integrity proof.

CPU and CUDA autocast must both be disabled in the execution thread; ambient
autocast is rejected before the forward. When a model exposes a
`config.use_cache` field it must be explicitly `False`, and a returned
non-`None` `past_key_values` is rejected. A model with no such field, or
arbitrary trusted model code, may still allocate hidden caches; the owner must
bind the model family and runtime configuration to a verified workload profile.
Detected changes or interruptions during registered-state checks poison the
worker. After a synchronized failed forward, the same best-effort state check
is performed before releasing the worker's model reference. If that check
detects a change or cannot establish the state, `WorkerStateError` preserves
the original classified failure diagnostic when available. The budgeted owner
retains the execution lease and quarantines its admission gate for this error.
This does not restore a model still held by another owner; reload or
independently verify it before reuse. A failed device synchronization takes
precedence over ordinary state and execution errors.

Cancellation is cooperative. It is checked before starting and after the
forward; an already launched CUDA kernel cannot be preempted. Device work is
synchronized before a result or a failure is returned, so a surrounding
device-memory lease must remain held until the call has returned. If device
synchronization fails, completion is unknown. The worker latches a
process-wide quarantine before releasing its execution lock, refuses further
worker forwards, and tries to retain known model, request, and output
references until process teardown. Retention itself may fail under memory
exhaustion; the quarantine remains in force. On every exception, the service
must inspect `worker.quarantined` as well as the exception type: a secondary
failure during unwinding can mask the synchronization error. If quarantined,
retain the device lease and promptly terminate the dedicated execution process.
The latch cannot stop unrelated
CUDA code in the process, and model internals may already have released other
temporaries; it is not a substitute for exclusive process ownership. Late
cancellation discards the result. Out-of-memory and unexpected execution
failures are reported explicitly and poison the worker: it drops its model
reference, and the broker must reject pending work and replace the worker.
The worker cannot release references held by other owners or guarantee that
the CUDA caching allocator returns memory to the operating system. Retrying
with a different workload is a separate, explicitly labeled run.

The worker's serial execution and per-request byte caps are not evidence that
multiple forwards overlap on the GPU. GPU throughput, peak memory, and
concurrent scheduling require separate measurements and an external admission
and memory manager.
