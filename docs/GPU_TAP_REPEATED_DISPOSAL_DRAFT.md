# Repeated tap owner disposal seam (CPU-only draft)

This private helper is not a GPU launcher or authorization to measure the
repeated-tap workload. It is a single-use part of a future, separately
supervised owner. That owner must establish the physical device/session and
the whole-run admission, use a reviewed exact workload and callbacks, retain
the helper itself before calling it, and terminate on **any** escaped error or
ambiguous return. It must never catch a disposal error and continue device
work, release claims, or report a clean attestation. A supervisor must keep
the unsafe marker until the process is reaped and a separately justified
recovery is complete.

The healthy path starts only when `model`, `converted`, `input_ids`, the
checked `default_stream`, and the
two CUDA event slots are occupied, all other owner resource slots are empty,
and `pending_return`, `failure`, and `cleanup_failure` are empty. The owner
must remove other references before starting disposal. Every named resource
must support a weak reference; a resource that does not is a fail-stop, not a
reason to skip the check. The model and converted-model slots may alias. The
dedicated owner must disable automatic cyclic garbage collection before the
disposal sequence, keep it disabled through process exit, and avoid explicit
global collection. This prevents a single collection from executing multiple
uncontrolled finalizers between custody checks; it does not control arbitrary
manual collection or callbacks supplied by trusted code. Disabling cyclic GC
does not suppress Python shutdown hooks or shutdown finalizers. Both successful
and failed owner exit therefore require a reviewed direct hard-exit path after
the appropriate flush/attestation decision; the supervisor must strictly reap
the child and validate the corresponding result.

The helper checks the same owner session, physical lease, admission, roots,
and runtime identity around each guarded operation. It performs one successful
synchronization with the named roots retained, drops one named slot at a time
while checking the remaining slot identities, and requires their weak
references to be dead before a second successful synchronization. Cyclic or
externally retained resources fail closed with the admission still charged;
they are not collected in this path. It then runs up to three trusted
cleanup callbacks, with checks between callbacks, synchronizes a third time,
and separately guards the allocated and reserved counter reads, requiring an
exact, type-strict zero for each. The fixed owner must make each supplied
callback contain only one CUDA-facing operation, or explicitly guard each
additional operation inside that callback.
Only then does it release the trace and whole-run admission claims and verify
the release; the physical device lease remains the outer owner's custody.
The fixed owner must prescribe and validate the actual CUDA cleanup operations
and zero-counter source; an empty callback tuple here does not establish that
the intended allocator or library cleanup occurred.

A failed or interrupted synchronization, cleanup, custody check, counter
read, or release stops this healthy sequence. Known slots and the admission
remain held until process exit except where release had already committed
after verified disposal. Unexpected callback return values are retained when
assigned. Failures during trace capture require independent rooting of capture
containers and the original private exception graph; session invalidation can
clear its event list, and a value created before event insertion may exist only
in the failing frame. Occupied failure or capture slots do not enter healthy
disposal. The caller must root and preserve those failure objects through exit.

These checks concern named Python objects and reported counters. They do not
prove physical CUDA quiescence, absence of hidden native allocations or storage
aliases, or correct behavior of arbitrary callbacks, finalizers, native code,
other processes, or a modified interpreter. An interruption between a native
return and its first Python assignment, or during an exception handler, may
leave an object unrooted. Python can also suppress an exception from a
finalizer. The trusted fixed owner, runtime and outer fail-stop supervisor are
therefore essential; the helper does not offer general asynchronous safety.
