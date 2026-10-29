# CPU plan-case prototype

`CPUPlanWorker` is a trusted, in-process prototype for executing a fixed sequence of ordered activation plans on one CPU model. It is **not** a broker, IPC service, GPU execution path, device-admission system, security boundary, or implementation of the pretrained causal protocol. The current forward-only broker does not accept these requests. No held-out inputs or study evidence are included here.

The trusted caller creates and controls the model, owns all case construction and output filtering, and excludes arbitrary model access, custom hooks, model-tree mutation, and concurrent model execution. All registered parameters and buffers must be on CPU, all modules must be in evaluation mode, process-global and pre-existing model hooks are rejected, and an exposed model cache must be disabled. A trusted model implementation must itself avoid CUDA, worker threads, unregistered state changes and side effects; inspecting registered tensors cannot enforce these properties. The prototype rejects ambient autocast and returned key/value caches. It uses a seeded CPU RNG scope and inference mode. These measures do not replace checkpoint hashes, runtime and precision metadata, deterministic-algorithm policy, independent baselines or experiment-specific state controls.

The worker does not validate a research protocol or exposure inventory. A future study custodian must bind the current protocol, exposure inventory and source identity before preparing candidates, and must filter private errors, logs and traceback locals as well as outputs. This prototype grants no permission to consume reserved study cases.

## Continuous ownership and ordered forwards

One case acquires a process-local `ModelLease` for the whole sequence. Each `ActivationPlan` uses a single restricted child scope under that exact lease, on the owner thread. The parent cannot be released while a child or its hook cleanup is active or uncertain. Cooperating tracing and intervention APIs therefore cannot acquire the model between case forwards. The case worker serializes its own cases, but does not provide fair queuing; a separately reviewed scheduler is required. The lease cannot prevent uncooperative direct calls or access from another process, and it provides no CUDA synchronization or device lifetime guarantee.

Each forward takes a batch-one, bounded CPU `input_ids` tensor and optional same-shape `attention_mask`, a seed, and a bounded ordered plan. Only one observed invocation per selected module is accepted. Literal tensor sources are copied and retained before use. A step can instead use `value_ref` to name a previous capture from the same case and identical input snapshot; `source_capture` retains its existing same-forward, same-site meaning. The worker records the input digest, seed, plan metadata and exact retained source references privately. The digest includes the input names, shapes, dtypes and bytes; it does not identify the checkpoint, code, RNG implementation, or runtime. The trusted protocol owner must save or reconstruct actual inputs and verify all required provenance.

For example, a trusted custodian may capture a clean vector, inject an explicit delta, verify the resulting raw bits, then restore the clean vector:

```python
from looking_glass.plan_worker import CPUPlanWorker

worker = CPUPlanWorker(model)  # trusted eval-mode CPU model, cache disabled
case = worker.case("development-case-1")
with case:
    site = {"module_path": "transformer.h.2.mlp", "batch_index": 0,
            "token_index": input_ids.shape[1] - 1}
    clean = case.run({"input_ids": input_ids},
                     [{"name": "clean", "operation": "capture", **site}], seed=17)
    changed = case.run({"input_ids": input_ids}, [
        {"name": "before", "operation": "capture", **site},
        {"name": "inject", "operation": "add", "value": delta, **site},
        {"name": "after", "operation": "capture", **site},
        {"name": "repair", "operation": "patch", "value_ref": clean.captures["clean"], **site},
        {"name": "repaired", "operation": "capture", **site},
    ], seed=17)
    case.require_added(changed.captures["before"], changed.sources["inject"],
                       changed.captures["after"])
    case.require_equal(clean.captures["clean"], changed.captures["repaired"])

export = case.export(changed.logits)
try:
    # Only a trusted consumer should see this mutable copy.
    logits = export.tensor
    # Use and release every tensor/view alias before closing the export.
    del logits
finally:
    export.close()
    case.discard()
worker.close()
```

`require_equal` compares raw tensor bits. `require_added` compares the exact retained pre-vector plus the exact retained source with the post-vector; both create bounded private temporary copies. A failed check invalidates the entire case even when the caller catches its exception. These checks do not establish that a causal hypothesis is true. Protocol-specific checks must still verify every required source, negative control, post-action vector, model identity and held-out evaluation.

## Custody and publication

Runs return opaque references, not tensors. A run's source references, capture names, plan metadata, failures and raw plan specs can reveal the hidden intervention location. All methods, including `private_metadata`, `private_failure`, `export_private_failure` and `active_exports`, are available only to the trusted custodian; they are **not** an investigator-facing API or a redaction boundary. A successful whole case permits `export` of saved captures and last-token logits. A failed case has no ordinary published results, but a trusted custodian can inspect its bounded private partial metadata and, if the worker is not quarantined, explicitly export completed retained tensors. A forwarded or previously returned object cannot be recalled, so the caller must not send private references, errors or metadata to an investigator.

One `HostTraceStore` caps aggregate logical bytes and item count for retained source snapshots, captures and logits across active and completed cases. `max_stored_cases`, `max_runs_per_case` and `max_steps` also bound the number of retained cases and recorded plan entries. The store remains charged until `case.discard()`, which is only allowed after a safe successful or failed case; a quarantined case remains retained for process disposal. Literal sources, prior captures and logits from prior forwards remain charged throughout the case.

An export is a **second**, separately charged copy. The worker reserves its `max_export_bytes`/`max_export_items` quota before copying and roots each `PlanExport` in `active_exports()` until its trusted consumer explicitly closes it. Export copying is serialized with active cases: a new copy is rejected while a case is active, and a pending copy blocks the next case. A completed export can remain in the consumer's hands while a later case runs. This accounting persists after `case.discard()` and permits recovery of an export handle if the caller is interrupted at a return boundary. A failed or uncertain export remains quarantined and cannot be closed as if disposal were proven. `PlanExport.tensor` is a mutable alias of the export's copy, never a store backing tensor. Before `close()`, the consumer must finish all operations and discard every tensor and view alias, including references in retained exceptions; otherwise the accounting can be released while memory is still live. The consumer must reserve a separate budget for any additional copies it creates.

The quotas bound declared logical payloads, not process RSS or physical peak memory. A current plan can hold its own bounded captures and sources alongside stored copies; control checks, input snapshots, full logits, model intermediates, allocator overhead and caller-owned objects add memory. `max_output_bytes` checks full logits only after the model has produced them; `max_tensor_bytes` similarly checks hook outputs after production. These limits are not a measured plan-specific resource profile and are insufficient for admitting GPU jobs. The caller must not mutate inputs or source tensors while they are being snapshotted.

## Cancellation and uncertain state

Cancellation is cooperative: an event is checked before a forward and again after the synchronous CPU forward, hook cleanup and state checks. It does not interrupt a running forward. A cancellation raised by model code during a forward, out-of-memory, unknown cleanup, an unexpected cache, uncertain model state or an asynchronous interruption poisons the worker and retains or quarantines ownership; it is not treated as a clean boundary cancellation. The previous completed private records remain distinguishable from the failed attempt. The original model may have changed RNG, buffers or unregistered state and must not be reused merely because a context exited.

The worker strips tensor-bearing inner execution traceback frames when propagating its normal classified failures. As with other Python context managers, arbitrary asynchronous interruption at a call/return or lock boundary is not universally recoverable. Retain the case object; if entry returned but the `with` body did not start, the owning thread may call `case.abort()` only after externally confirming quiescence. An interrupted exit can report `case.status == "recovery_required"`; after its stack has unwound and quiescence is established, the owner calls `case.quarantine_interrupted_exit()` to poison the worker and retain the case. No next case is admitted while that exit marker remains. If the exit completed before its return was interrupted, `case.status == "finished"` records the completed outcome. If a forward, hook cleanup, lease release, or output copy is uncertain, retain the associated claims and discard the worker/process instead of declaring a clean completion. Recovery APIs and private errors must remain on the trusted side.

Before any GPU or broker integration, the system still needs a reviewed typed custody handoff, an exact measured plan-specific host/device profile, shared exclusive model and device admission, device synchronization and cleanup through lease release, fair scheduling, and cancellation/disconnection handling. This CPU prototype does not supply those guarantees.
