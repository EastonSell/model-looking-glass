# Synthetic GPU worker failure validation

This one-shot experiment is enabled for a separately coordinated device slot;
no real-device result is claimed by its source. Its model is a randomly
initialized one-layer GPT-2 with a two-token input; it does not download a
checkpoint and is not a throughput or model-quality benchmark. A cooperative
supervisor and physical-UUID device lease must cover the complete process.
The PyTorch native allocator is capped at 512 MiB, and admission retains at
least 2 GiB of device-wide headroom. The workspace estimates are accounting
commitments, not measurements or a hard limit on native allocations.

The first phase directly exercises three separate one-event trace sessions:
success, a deliberately raised post-forward trace cancellation, and success.
Each capture is at most 64 bytes. The completed event remains privately rooted
until synchronization; the capture, input, model-output object and logits are
dropped and weakref-checked before a second synchronization and release of
the trace and workspace claim. This instrumentation is a
synthetic direct trace exercise, not support for tracing in the fixed service.

The next phase uses the actual fixed `LifetimeWorkspaceWorker`, with zero
trace claims. It runs a baseline request, accepts a cooperative cancellation
after the model call returns and before the worker's guarded synchronization,
and then runs another successful request. Cancellation is not GPU preemption.
Before each following safe checkpoint or request, the single producer waits
for the exact broker's pending, running, preparing and job-root inventory to
be idle; timeout or an unhealthy owner ends the process without another CUDA
call. This point does not prove disposal of outside aliases.
The final request performs the same small forward and deliberately raises
`torch.OutOfMemoryError`; it does not exhaust memory or establish recovery from
a real allocator OOM. The public outcome can instead be a synchronization
failure. Its private original exception and both accounting leases remain
rooted, and the unsafe authority rejects a subsequent request without a forward.

Every saved allocator/free-memory checkpoint is taken after a guarded device
synchronization and is a point observation. There is no CUDA query or cleanup
attempt after the authority becomes unsafe. The owner leaves the canonical
unsafe marker and exits nonzero; a separate qualified recovery must use the
saved owner, supervisor and process evidence. A clean result or native CUDA
cleanup must not be inferred from process exit or marker removal.

The private output is bounded by a fixed inventory of records, each at most
64 KiB. The controller records the existing prelaunch and post-exit Windows
Job counts; those samples do not prove CUDA cleanup. The allocator cap,
readback, runtime and model size are owner self-reports rather than independent
hardware attestations. The operator must discard or separately bound inherited
stdout and stderr.
It requires at least 2 GiB of free disk immediately before launch. Source and
dependency binding assume trusted local files and cooperative participants.
Python/native temporaries interrupted before assignment, external revocation
between point checks, other GPU processes, and native process teardown are not
fully observable. The CPU tests use the real tiny model and worker with CUDA
hidden; they do not establish a real-device outcome.
