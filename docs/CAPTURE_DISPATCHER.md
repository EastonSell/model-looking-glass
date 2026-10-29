# Request-local batched capture primitive

The dispatcher is a synchronous, single-use building block for a future
shared-model worker. It slices a normalized activation tensor for several
requests after one forward call. It is not a scheduler, GPU lease, model
pool, batching service, or proof that concurrent model execution is safe.
The existing worker must not accept arbitrary client-supplied hooks or code.
Integration must be done by its trusted model owner under an exclusive model
lease and validated separately.

## Ownership contract

The worker creates one immutable `CaptureRequest` per request. Each request
includes a run, job and request ID, a forward index, an optional generation
step, per-request event/byte caps, and one or more `SampleSelection`s. Each
selection gives the original batched output row, a request-local sample ID,
and output-local token positions. Every batch row must have exactly one owner;
request IDs must be unique within a batch and sample IDs within a request.
Requests with the same run ID must identify the same forward index within a
batch. Different run IDs may have different forward indices.
Duplicate or missing owners, IDs and invalid token positions are rejected
before executing the forward. Selected token positions are zero-based indices
within the hook output: they are not absolute sequence positions, cache
coordinates or a claim about prefill versus decode.

The trusted owner installs one forward hook per observed module and removes
every hook in a `finally` block, including on registration, forward, or
capture failure. Normalize that architecture's output before `record`: for
example, GPT-2 block and attention outputs are tuples whose first element is
the hidden-state tensor, whereas the MLP output is a tensor. This extraction
must return `None` from the hook and never replace or mutate the model output.
Hooks cannot be shared with another session or attached to other model calls
while the dispatcher runs. The caller then supplies exactly one synchronous
forward callable to `run` and calls `record` from that forward's owner hook.
Pass the output selector when known: `()` identifies a raw tensor output,
`(0,)` identifies the first tuple/list item, and `None` records an unknown
selector. Record one selected tensor per observed module invocation; separate
outputs from the same invocation are not supported by this primitive.

The accepted tensor is a materialized, dense, strided floating `torch.Tensor`
(not a tensor subclass or a meta tensor) shaped
`[batch_size, sequence_length, ...]`, with nonzero trailing dimensions; the
first two axes must exactly match the declared batch and sequence lengths.
Nonfinite values are preserved without checking. Experiments requiring finite
activations must validate them separately. The primitive supports direct
capture only; it does not implement gradient readout, autograd sharing,
interventions, generated cache tracking, or checkpoint recomputation.

Each result is an independently owned detached sample slice; saved values do
not alias the model output or another event. Each carries run/job/request and
sample IDs, batch/token positions, forward and optional generation-step
metadata, module path/kind/layer, invocation count per module within this
forward, and a sequence number unique within the run. Original and saved
shape/device metadata distinguish source from retained snapshots. An event is
observation metadata, not a causal explanation.
`CapturedSlice.to_trace_event()` returns a validated, tensor-free version 1
trace-schema event; its saved shape has one fewer dimension because each slice
selects a single source batch row. Generation stage and cache fields remain
unknown unless a future integration supplies independently verified context.
Conversion rejects a payload whose shape, dtype or device has changed since
capture. It does not detect in-place changes to values: callers must preserve
the snapshot if they use the metadata as a record of what was captured.
The result may be empty when no selected hook fires; an empty trace is not
evidence that an unobserved computation did not occur. An integration that
requires particular modules to execute must validate coverage explicitly.

The serial owner must explicitly supply `sequence_starts` for every run ID,
including zero for a new run. For a run ID used in more than one dispatcher
call, it must assign and persist the next sequence number before the next
call. The dispatcher guarantees uniqueness within a call but does not
coordinate starts across workers or processes. Starting again from zero for
the same run ID would create ambiguous event identities. Alternatively,
assign a distinct run ID to each call.
For a reused run ID, the owner also advances `forward_index` for each distinct
root forward. Neither cross-call sequence nor forward-index uniqueness can be
verified by an individual dispatcher instance.

## Limits and failures

Per-request and whole-batch event counts and payload-byte limits are checked
for all slices of an event before making any slice or device-to-host copy.
Limits cover retained tensor payload only. They do not bound model weights,
input/output activations, temporary slices, transfer buffers, allocator
reservation, Python metadata, or tensors retained by the caller from earlier
runs. The worker must separately budget host and device memory, reserve
headroom, manage lifetime across requests, and handle CUDA out-of-memory.
The `cpu` capture policy performs synchronous copies to host; `source`
retains each snapshot on its source device and therefore consumes GPU memory
for CUDA outputs. No pinned or asynchronous transfer is used.

The entire batch is transactional: an overflow, malformed record, wrong
thread, or failed forward clears all internally held snapshots and yields no
results. A hook failure poisons the run even if model code catches it.
Exceptions do not carry retained snapshots in dispatcher state; external
tracebacks and model code may still hold their own tensors. Successful
results transfer ownership to the caller, who must release them. Retrying is
a new run and must not silently change its capture configuration.

Dispatch rejects an active asyncio event loop, asynchronous/generator forward
callables or results, cross-thread recording, reentrant recording, and
overlapping dispatchers in the same process. This conservative restriction is
not interprocess coordination or a substitute for the future GPU lease.
The worker must own forward execution and hook lifetime; the dispatcher cannot
prove that unrelated model calls were excluded. Deferred-work detection only
inspects the top-level callable and its return value; the trusted owner must
not hide futures or coroutines inside containers and must ensure device work
has completed before reporting success. A CUDA kernel cannot be
preempted through this API; cancellation belongs to the worker and partial
results must not be released.

These boundaries currently have CPU unit tests with a small repeated-module
model. They have not established numerical equivalence for batched pretrained
inference, cancellation under CUDA, GPU memory limits, throughput, or safety
of simultaneous model execution.
