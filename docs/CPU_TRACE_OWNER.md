# Private CPU trace owner (prototype)

`CpuTraceOwner` coordinates one trusted `TracedGPT2Worker`, one producer
envelope and one complete private batch at a time. It admits the producer and
all child source claims before starting the forward, stages all expected
slices, and exposes tensor-free metadata only after source disposal and a
complete batch seal. It is **not** connected to `Broker`, IPC, CUDA, a
scheduler, or a client transport; it cannot retrieve or send the stored trace
payloads. There is no multi-job co-batching, fairness guarantee, physical RSS
limit, or authentication boundary between Python code in the same process.

The owner, worker, model, plans, claims and registries are cooperative private
capabilities. The trusted caller must exclude other code from mutating or using
the worker/model, specs, input tensors and child capabilities during the job.
Plan/profile fields are copied and revalidated at registration; the caller must
not mutate those objects while the copy itself is being constructed. The worker
must use the standard unmodified GPT-2 and module forward implementations,
with no monkeypatching or hidden side effects. Shape checks are not a proof of
code or weight integrity: altered code can allocate a different output despite
unchanged weights. The owner must also freeze the complete model configuration,
including attention and workspace-affecting settings; the worker's state
snapshot and this owner's preflight inspect only part of that configuration.
`model_id` is an owner label, not a weight fingerprint.

The actual worker must be CPU float32, in evaluation mode, with cache,
auxiliary hidden-state/attention outputs and ambient autocast disabled. The
owner verifies the head shape and checks that the expected raw logits for the
declared batch and sequence fit the worker's output limit **before** execution.
This assumes the trusted standard model implementation. The worker's own model
lease and postchecks still apply; a poisoned worker cannot be reused.

Each caller must reserve its original inputs **before allocating them**, in a
separate `HostRequest`/`HostClaim` from the same `HostBudget`. The claim must
cover the sum of the reported input storages and their item count. Inputs must
be plain, contiguous CPU tensors occupying their entire reported PyTorch
storage, and must stay stable through the forward. These checks do not discover
a larger external Python or `frombuffer` backing owner: the trusted caller must
provide truly standalone inputs or separately charge every hidden backing
allocation. The owner pins the input claim while the forward can use it and
does not release the caller's claim. The caller remains responsible for all its
own aliases after execution and for releasing that claim only after disposal.
The input operation label includes the job ID, registration ID and generation
for supervised recovery; the exact ledger/request generation and owner/process
scope must still be checked, and labels alone do not authenticate an operation.

The producer profile is an operator-supplied logical commitment. The checked
minimum is twice the worker's input byte cap and eight input items (two
snapshots of up to four named tensors can overlap), twice the output cap and
two output items (raw logits and the CPU output copy overlap), and the plan's
declared trace byte/event caps. Workspace must be positive, but **no workspace
estimate or physical memory peak is validated**. The operator must justify a
worst-case workspace including model attention/activations, intermediates,
dispatcher index tensors, finite checks, metadata, allocator overhead and any
other live owners. Resident model parameters and buffers span jobs and are not
included in this per-job ledger; their owner must reserve them separately.
Producer and delivery-source claims overlap intentionally;
each event also reserves child metadata (currently 65,536 logical bytes), and
store staging reserves its own copy. Manifest copies and temporary Python/JSON
metadata are outside these claims. These claims are not allocation interception
and cannot by themselves bound process memory. Admission is sequential and
nonblocking, not an atomic whole-job reservation: contention or a store-copy
limit can cause failure after some claims were acquired. Do not wait for more
capacity indefinitely while holding a partial bundle.

The trusted caller records `(broker_id, process_generation, job_id,
registration_id, generation)` and the exact owner instance before calling
`register`; generations must increase within that owner. After an interrupted
registration, inspect `lookup` with the exact already-recorded key before any
retry. Different owner instances and process generations are separate scopes;
the trusted broker must not reuse a job ID during its lifetime. Schema event
identity is `(run_id, sequence)`, independent of the broker job ID: an external
allocator must ensure unique run/sequence ranges and correct forward indices
across all producers. Token positions refer to the current hook output, not
absolute generation or cache coordinates. A supplied generation step is a
trusted declaration; cache position and generation stage remain unknown.

The intended lifecycle is:

1. Construct the owner and bind a one-job plan to every row; register the job
   with the private plan/profile and known identity, then recover it by `lookup`.
2. Call `prepare()`. Only after it returns may the trusted caller submit the
   exact `ForwardRequest`, original-input request and input claim to `execute()`.
   `execute()` returns no logits or trace tensors. The source bundle and all
   producer claims remain rooted through validation and private staging.
3. After successful staging, call `discard_result()`. Before calling
   `confirm_producer_disposed()`, the caller must establish that **all** worker
   result, source, output, workspace, temporary, traceback and exception-graph
   aliases covered by those claims have actually gone, or are separately
   charged. This is a trusted attestation, not a reference-count check. Tensor
   destruction happens outside the job's state lock. A saved exception or a
   custom exception attribute can retain tensors even after the method returns.
4. Call `seal()` and then `manifest()`. The complete manifest binds each child
   identity and full trace event to its expected row/site/sequence. No partial
   manifest is returned. Call `dispose(timeout=...)` and then `owner.retire(job)`
   to drain the private staged copies. Only after all caller input aliases are
   disposed may the caller release its independent input claim/request.

`cancel()` sets a prompt local signal and coordinates with the short commit
lock, then signals the batch outside that lock. Cancellation cannot preempt a
running synchronous forward or copy. A cancelled job that has not started a
forward uses `abort_before_forward(timeout=...)`; this also recovers known
partial-admission failures by their exact rooted child keys. A normal drain
timeout returns `False` with claims retained; retry the same call. The timeout
only limits child drain waits: synchronous cancellation itself can block.

When an ordinary error or cancellation has a verifiably finished or unstarted
producer and an intact or completed input pin, the job enters `failed`. It never exposes a
manifest. After the caller drops the entire exception/traceback graph and all
other source aliases, it can call `discard_result()`, explicitly attest with
`confirm_producer_disposed()`, and `dispose()`/`retire()`. Out-of-memory
failures leave the worker poisoned even if known host resources are drained.
If pin start/finish, producer completion, child construction, or cleanup is
ambiguous, or an asynchronous interruption occurs, the job conservatively
enters `quarantined`. No public method guesses that its claims are safe to
release; a trusted supervisor must establish quiescence and reconcile the
exact retained capabilities, or replace the affected process. General
asynchronous interruption of Python lock exit can strand a lock and requires
process-level recovery. A process-lifetime supervisor must retain strong
references to the exact owner, budget, job and affected capabilities until
reconciliation or verified process death; this prototype has no global
quarantine root, and dropping those references is not disposal evidence. A
raised call can also have committed immediately
before its return was interrupted: inspect the already-rooted exact job/state
instead of assuming the call had no effect. The same applies to a seal whose
return was lost. A manifest read that began before cancellation may complete
after cancellation; a future transport must enforce its own late-cancel and
delivery-custody rules.

Preflight errors clear ordinary nested traceback frames and owner-local input
references on a best-effort bounded basis. Custom exceptions, oversized groups,
caller frames and exception attributes may still retain inputs. A caller must
not treat an error, timeout, cancellation, or ledger counter as proof of actual
storage disposal. This prototype has only been exercised on CPU; it provides no
GPU admission, cancellation, out-of-memory recovery, or throughput evidence.
