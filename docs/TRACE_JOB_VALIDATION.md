# Complete CPU GPT-2 trace-job validation

`make_trace_job_layout` derives expected event slots for a single trusted job
from a previously bound `BoundTracePlan`. It is a **metadata and payload
validation primitive**, not a worker, scheduler, publication seal, memory
reservation, transport, or causal explanation. It is not connected to the
current broker or IPC service. The trusted model owner supplies the actual
batch size, output-local sequence length, GPT-2 hidden width and layer count;
this function does not inspect or authenticate a model, job, input or team.

The current layout is intentionally narrow: standard CPU GPT-2 float32
hidden-state outputs, one invocation of each selected attention, MLP or block
site, and one selected slice for every sample at each site. It derives the
completion-hook order for ascending layers: attention, then MLP, then the
enclosing block output, with samples in batch-row order for each site. A
caller may list sites in any order. Each slot has the expected original row,
sample ID, ordered local token indices, module path, layer, output selector,
invocation and exact payload bytes. `max_slots`, `max_event_bytes`, the
request's event cap and its trace byte cap are checked before execution;
tuple lengths are checked before reconstructing selection objects. These are
declared payload limits, not host or device memory admission.
The owner must not mutate the bound plan during layout construction; after
construction, later edits to caller-owned plan or exported slot objects do not
change the private layout. Arbitrary same-process code remains outside this
cooperative contract.

`layout.validate(events)` accepts an exact tuple of `CapturedSlice`s only.
It rejects missing, extra or duplicated slots; wrong request/job/run/sample
identities; mismatched coordinates, GPT-2 module paths or tuple selectors;
and a sequence that differs from the expected slot's position in this fixed
hook order. It also requires ordinary materialized, contiguous, detached,
float32 CPU tensors with the declared shapes and finite values. Finite checks
process at most 4,096 elements at a time, but their temporary tensors and
Python metadata still require separately reserved workspace. The output is a
complete tuple of tensor-free `TraceEvent` metadata, ordered to match
`layout.slots`; the input events can arrive in any order. To stage the actual
event objects, match by `(batch_position, layer_index, module_kind)`: never zip
an arbitrarily ordered input tuple with the slots or returned metadata.

The validation checks the consistency of **declared** metadata. It cannot
prove the values came from the stated input, model or site, or that the
owner-supplied generation step is true. Token positions index the local hook
output, not necessarily absolute generation or cache coordinates. Conversion
leaves generation stage, cache position and cache usage unknown. The trusted
owner must bind the actual worker and inputs, verify model postchecks and
coordinate unique `(run_id, sequence)` and forward IDs across all producers.

Each reported tensor storage must cover exactly its logical payload and the
reported memory-address spans must not overlap one another. This detects
overlap among these spans; it **does not** prove the tensors own their backing
allocations or that unreported aliases do not exist. For example, a small
`torch.frombuffer` tensor can retain a much larger external Python buffer.
The worker must produce trusted standalone dispatcher snapshots or separately
account for every retained external owner, and must maintain exclusive stable
source ownership through validation and staging. Values are mutable: a
successful check does not prevent later mutation.

This helper does not reserve source, caller input, logits, model workspace,
per-slot metadata, store-copy or transport memory. A future seal must reserve
all needed overlapping claims before each allocation, hold them until every
operation and alias has ended, and publish one complete manifest atomically
against cancellation. On failure the validator clears its own event argument,
attempts to strip traceback/context frames, and reraises the original exception
type. It handles children of exact built-in exception groups only, with a limit
of 128 exceptions and 16 levels. Oversized groups, custom exception classes,
exception arguments or other attributes, caller frames, and other aliases can
still retain tensors. Failure must not be used as evidence that source storage
was disposed. The owner must keep claims charged until it establishes disposal
or quarantines the uncertain job.

The CPU tests use dispatcher-produced slices and an offline randomly
initialized tiny GPT-2 to check site order and metadata. They do not establish
pretrained-model behavior, multi-team co-batching, GPU correctness, memory
enforcement, IPC custody, throughput or causal localization.
