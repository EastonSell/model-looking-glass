# Compatible forward microbatch primitive

Status: **CPU assembly and result splitting only**. This component does not
run a model, select work from a queue, acquire a GPU lease, transfer tensors,
authenticate clients, or demonstrate concurrent GPU execution. It is not
wired into the broker or resident worker. It prepares a possible future
batch of compatible eval forwards and fails closed for the unsupported cases
described below.

## Trusted admission requirements

Only a trusted service may construct a `ForwardContract` and `BatchProfile`.
Strings such as `owner`, `model_id`, `checkpoint`, `runtime_id`, and
`certification_id` are metadata, not credentials or proof of identity. The
broker must authenticate job ownership, generate unguessable job IDs, validate
an immutable content-addressed checkpoint and adapter identity, and bind the
request to the actual resident model and runtime; a moving tag alone is not an
immutable checkpoint identity. `runtime_id` must identify the relevant backend, software,
precision, determinism and numerical settings. The plan does not inspect model
weights, hooks, global settings, or the CUDA device.

The profile must refer to evidence for the **whole combined workload** on the
selected model, runtime, device, input shape, and batch size. It explicitly
requires an eval forward with no autograd capture, stochastic/RNG use,
interventions, trace hooks, input/output KV cache, or mutable cross-request
state. A trusted adapter must actually enforce these properties, including
disabling model-specific caches such as `use_cache`. Matching declarations do
not enforce them.

The profile must also certify **batch separability**: each row's outputs depend
only on that row and the immutable model. Eval mode and deterministic execution
alone do not prove it; a model subtracting a batch mean is a counterexample and
could expose another client's data. Validate batched outputs against serial
reference outputs on independent, representative held-out inputs under the
profile's preregistered `serial_atol` and `serial_rtol`. Retest when the model,
adapter, runtime, device, dtype, numerical settings or workload changes. The
fields record tolerances; this primitive does not run that certification.

Every request must match the profile's model/checkpoint/adapter/runtime, dtype,
device, seed and determinism contract. Seeds must be integers in `[0, 2**63)`,
matching the resident worker. The inputs must be CPU, rank-2 integer or
boolean tensors with exactly the declared names, dtypes, and shapes. Supported
names match the initial resident worker: `input_ids`, optional
`attention_mask`, `position_ids`, and `token_type_ids`. Outputs must be one
finite, detached CPU logits tensor shaped `[combined_rows, sequence, vocab]`,
with the declared dtype and output width. Different shapes, unsupported input
names, tracing, and interventions are rejected rather than silently altered.
The input container must be an exact `dict` or an exact read-only
`mappingproxy`, as used by resident-worker snapshots; custom mapping
subclasses are rejected as direct inputs. A proxy can wrap a custom or mutable
mapping, and does not freeze the backing mapping or its tensors. Only proxies
from trusted plain-dict snapshots are supported; the broker must protect the
backing mapping as well as the tensors throughout snapshotting. The container
type check does not establish this provenance.

Shape and dtype do not validate token, position, segment, or mask values. The
trusted adapter must validate model-specific domains before grouping, or
explicitly accept that one malformed input can fail the whole shared forward.
The service must own or lock each request's tensors while the constructor
validates and snapshots them; copying a tensor concurrently with client
mutation is not atomic. Once construction completes, the plan no longer uses
the caller's input storage.

## Limits and fair scheduling

`max_requests`, `max_per_owner`, `max_batch_rows`, and `max_token_positions`
bound the requested composition. `max_per_owner` is a per-batch cap, **not** a
fairness scheduler; the broker still has to select from its fair per-team
queue without bypassing waiting teams. The plan consumes at most
`max_requests + 1` items from an iterable before rejecting an oversized batch.

The `max_logical_input_bytes`, `max_logical_output_bytes`, and
`max_logical_total_bytes` limits bound logical tensor payloads, not peak host
RAM or VRAM. For example, snapshots coexist with concatenated inputs, and full
logits coexist with split output clones; finite checks and model execution add
more temporaries. Caller-owned buffers and plan results can persist too. A plan
retains its returned tensors until the plan itself is released, and these caps
apply per plan, not across plans or over time.

Before any GPU transfer or work, the broker must acquire a **whole-batch**
lease based on a trusted measured/conservatively budgeted workload profile,
including weights, activations, runtime allocator headroom, host buffers,
offload and actual input/output lifetimes. Do not infer the required GPU memory
by summing per-job estimates or equate the logical caps with available VRAM.
The broker must also check current device-wide free memory and its configured
reserve. `approved_rows` lists exact combined row counts covered by the trusted
profile; this module checks membership but does not validate the evidence.

## Lifecycle and ownership

1. Construct `Microbatch(requests, profile)` from already fair-selected jobs.
   The constructor validates the full planned composition before copying CPU
   inputs. Each `job_id` is unique and each `(owner, request_id)` pair is
   unique; job IDs should be broker-generated rather than client-chosen.
2. `cancel(job_id)` before `launch()` removes that input snapshot. `launch()`
   concatenates remaining requests in original order and returns the combined
   inputs, job IDs, profile ID, and a fresh `launch_id`. If all were canceled,
   it returns `None` and `outcome()` reports their cancellations. If the
   remaining row count lacks an approved profile, no forward is launched;
   canceled jobs remain canceled and the remaining jobs fail as
   `unprofiled_batch`. A separately specified run may be resubmitted later.
3. After `launch()` returns, cancellation suppresses delivery to that job but
   does not remove its rows or stop the shared forward. Cancellation, launch,
   and completion are serialized by the plan's lock. A cancellation racing
   completion succeeds only if it acquires the lock first; otherwise it returns
   `False` and the completed outcome remains unchanged.
4. The trusted worker executes under its immutable, no-grad contract, waits
   for all relevant device work and CPU offload to complete, and calls
   `finish(cpu_logits, launch_id=...)`. The broker must bind the worker request
   to the launch ID and verify that the returned worker request ID matches
   before supplying logits. The plan validates provenance token,
   shape, dtype, finiteness, and absence of autograd history, then creates an
   independent ordinary CPU tensor for each noncanceled owner. Results are
   returned in original request order with their job, owner, and request IDs.
   A launch ID is a correlation nonce, not authentication or proof that an
   arbitrary caller supplied the right model output.
5. If execution fails, call `fail(error, launch_id=...)` only after accounting
   for device work and cleanup. An OOM marks every noncanceled member failed;
   canceled members remain canceled. Invalid/stale output and execution errors
   also fail the entire noncanceled batch without delivering partial logits.
   The resident worker's typed `WorkerOutOfMemory` is classified as
   `out_of_memory`; a different wrapper exception remains an execution failure
   unless the trusted caller preserves the typed cause explicitly.
   The outcome requests worker quarantine after any post-launch failure. The
   broker owns synchronization, worker recovery, memory release, and recording
   the failed run; the plan neither retries nor declares a GPU safe to reuse.

Prelaunch concatenation or nonce failures leave terminal failed results and
re-raise; because no forward was issued, they do not request model quarantine.
Interrupts are re-raised after recording terminal outcomes or latching an
unpublishable state. The returned launch and outcome objects belong only to
the trusted broker; it must route each result
to its authenticated owner and prevent access to other owners' records. This
primitive does not itself provide a security boundary or ensure secure erasure.

If validation, allocation, or snapshotting fails during construction, no plan
is returned. The broker must retain the original selected-job identities and
finalize each affected request itself, including after interruption or host OOM.

If allocation or interruption prevents cancellation bookkeeping, failure
classification, construction of per-request records or the aggregate outcome,
or completion cleanup, no per-request terminal outcome is published. A failed
`cancel()` call does not confirm that cancellation was accepted. The plan
latches `publication_failed`, releases its owned input snapshots, and rejects
further launch, completion, failure and cancellation attempts. When
`publication_quarantine_required` is true, a launch was handed off or a worker
failure already required quarantine. The broker must preserve the selected
member identities, cancellation states, and accounting independently, wake
waiters with an unknown/unavailable status rather than inventing outcomes, and
quarantine as indicated, while retaining the lease until any in-flight worker
has completed and cleanup has been accounted for. A secondary failure while
recording a worker failure retains the original exception as its cause.
Under persistent host OOM, even creating new failure
records cannot be guaranteed; this primitive cannot perform broker recovery.

## Evidence and next steps

The accompanying CPU tests exercise a small deterministic eval module, compare
serial and batched logits with fixed `atol=1e-7`, `rtol=1e-6`, verify unchanged
model state and preserved preexisting parameter gradients, and test ownership, limits, cancellation,
stale tokens, invalid outputs, OOM and exceptions. This does not certify another
model or any GPU workload. Before broker integration, independently test
profile selection, fair scheduling, whole-batch leases, CUDA synchronization,
model immutability and batch separability on the actual target model and
hardware. Compare shared-service throughput and isolated latency under their
separate protocols; batching requests is not evidence of simultaneous GPU
execution or a speedup.

The existing broker currently executes one request at a time. Integration needs
a fair bounded grouping/dequeue policy, broker-generated batch/member IDs, an
exact whole-batch profile and lease, a worker configured for the combined row
count, and authenticated result demultiplexing. Do not pass one member's cancel
event as the shared worker cancellation event: that could discard every member's
result. An integration test must count actual worker/model invocations and show
one combined forward for compatible jobs; an isolated plan test cannot do so.
The current fair queue does not expose an atomic compatible-head grouping
operation; popping incompatible work and appending it again can change per-team
FIFO order. A bounded fair selection/staging API is needed before integration.
The budgeted worker also requires a service-verified 64-hex model fingerprint;
descriptive model labels do not meet that binding requirement.
