# Ordered activation plans

`ActivationPlan` performs several explicitly ordered captures and interventions during one model forward. It is intended for controlled experiments, such as injecting a known delta, recording the resulting activation and patching a candidate site. An observed response tests that intervention under those inputs; it does not by itself identify a general mechanism. Freeze the hypotheses, candidates, predictions, controls and held-out cases in a versioned experiment protocol.

```python
from looking_glass.intervention_plan import ActivationPlan

steps = [
    {"name": "inject", "operation": "add", "module_path": "transformer.h.0.mlp",
     "invocation": 0, "batch_index": 0, "token_index": 3, "value": delta},
    {"name": "after_injection", "operation": "capture", "module_path": "transformer.h.0.mlp",
     "invocation": 0, "batch_index": 0, "token_index": 3},
    {"name": "candidate_patch", "operation": "patch", "module_path": "transformer.h.1.mlp",
     "invocation": 0, "batch_index": 0, "token_index": 3, "value": clean_vector},
]
with ActivationPlan(model, steps, max_tensor_bytes=8_000_000,
                    max_capture_bytes=1_000_000, max_source_bytes=1_000_000) as plan:
    output = model(input_ids)
observed = plan.captures["after_injection"]
events = plan.metadata["steps"]
```

The example assumes a matching GPT-2 model and explicitly aligned vectors on the correct device and dtype. Verify model/checkpoint identity, input pairing, token alignment, RNG, model mode, cache and mutable buffers in the experiment protocol. The plan itself does not restore model buffers or RNG after a failure.

## Ordering and coordinates

Each step needs a unique `name`, `operation` and `module_path`; `invocation` defaults to zero. The supported operations are `capture`, `patch`, `zero`, `add` and `null`. Add requires a fixed external tensor `value`; patch accepts either such a value or `source_capture` naming an earlier capture at the exact same module object, invocation and batch/token coordinates. This explicit same-site reference enables a self-patch control. It cannot refer to a different site or a later capture. The other operations accept no value. External source tensors are detached, validated and copied at construction; a referenced in-plan snapshot is detached and counted once in the capture budget.

A self-patch with the current value preserves the selected tensor values, but its detached source cuts the gradient through the patched coordinates. It is therefore not a training-gradient no-op. Use the `null` control when gradients through the selected activation must remain unchanged, and compare against an untouched baseline to detect layout-sensitive effects.

The declared list must match **module-output hook completion order**. A child module's hook completes before its enclosing parent's hook. Consecutive steps targeting the same module object and invocation run in list order on the evolving output; a capture between add and patch records the post-add, pre-patch value. Steps for that same site cannot be split by another site. Actual site execution out of order, missing steps, repeated root forwards, reentrant targets and target calls that fail after the plan's pre-hook invalidate the plan, including failures caught inside the model. Invocations count starts observed by the selected object's pre-hook, from zero in that forward, including calls that are not selected by any step; aliases of the same object share that count. Other local or global hooks and custom calling implementations that can run before or bypass the pre-hook are unsupported: they can hide calls, and the plan does not detect or police them.

When both `batch_index` and `token_index` are omitted, the operation selects the whole tensor output. When both are present, they select a hidden vector from `[batch, token, hidden]`; positions refer to the current forward, not absolute generation positions. The supported module result is a tensor or an ordinary tuple whose first element is a tensor. Tuple fields other than the first are preserved. Metadata includes the selector (`[]` or `[0]`), actual step order, preceding same-site step names, shape, dtype, device, lookup path and all registered aliases. A forward hook cannot identify which Python call site invoked an aliased module; `call_site_known` is false.

If a candidate site executes before an injection site, its patch occurs first in time and cannot retroactively repair a later activation. List the true execution order and interpret the resulting intervention accordingly. Mutating steps clone outputs and may change strides, storage or aliasing; even a numerically unchanged null can affect layout-sensitive downstream behavior. A null control does not replace an untouched baseline, random-location controls, alternative sources or held-out evaluation.

## Ownership, limits and failures

The plan uses the shared process-local `ModelLease` and requires exclusive model ownership. Enter the plan, run the forward and exit on the same thread; a foreign-thread callback or active-context exit invalidates it. Thread-parallel and distributed model execution are unsupported. Thread checks are defense in depth: the owner must ensure no noncooperating thread can already be inside the model or dispatch a callback through publication. A late callback can invalidate subsequent access, but cannot recall tensors already returned. It cannot safely run alongside independent tracing or modifying hooks on the same model. It does not implement request-local concurrent interventions. A broker must hold its own device lease through CUDA synchronization and cleanup; the Python hook lease is not a GPU scheduler or a cross-process lock.

`max_tensor_bytes` limits each full module output or supplied source tensor; `max_capture_bytes` limits the aggregate retained snapshots; `max_source_bytes` limits the aggregate copied sources. `max_steps` and `max_invocations` bound plan size and selected-module calls. These bounds do not cover total peak device/host memory, temporary tensors, allocator reserve, output files or queue storage. Checks for finite values can synchronize CUDA; per-hook tree verification and all copies are part of the overhead to measure. The caller must also enforce workload, time, memory and artifact budgets.

Captures and metadata become available only after every step, the forward and context exit succeed. On failure, partial captures are not published; a quarantined plan may retain private buffers until its worker is discarded. If removal of a returned handle fails, the plan retains its lease and `plan.close()` can retry. If registration fails before a valid handle is returned, a hook may have been installed invisibly: the model remains quarantined and recovery requires a newly constructed model or worker. Do not copy a contaminated model. The caller must record failed runs and handle device synchronization and out-of-memory recovery separately.

`plan.captures` returns a new dictionary whose values are the retained, mutable snapshot tensors. Copying the dictionary does not copy those tensors; mutating a value changes later reads. Make and budget an explicit tensor copy if independent ownership is needed, and hash the actual saved artifact with dtype, shape and coordinates recorded by the experiment.

Python can interrupt context entry at its return boundary after hooks have been installed but before the `with` body starts. Retain the plan object: once the owning thread has established that no forward or noncooperating callback is active, `plan.abort()` discards the entered plan and removes its hooks. A foreign abort invalidates the plan and cannot release its lease. Exit, abort and close refuse to release while a forward may still be active. PyTorch can skip completion hooks on `KeyboardInterrupt` or `SystemExit`, leaving that state uncertain even after the stack unwinds; the original exception propagates, the lease remains held, and the worker must be quarantined and discarded. No Python context manager guarantees atomic recovery from arbitrary asynchronous interruption.

Activations and metadata may contain sensitive data. In a blinded injected-change study, raw plan specs and `plan.metadata` can reveal the hidden injection location; only the custodian should see them until the blind is lifted. Publish only the protocol-permitted, filtered artifacts.

`tests/test_gpt2_plan_reference.py` is an optional CPU integration fixture using fixed artificial tokens and a freshly initialized six-block GPT-2. It compares plan actions and logits with independent serial hooks and includes restoration, upstream and self-patch controls. This is software validation on a random model, not evidence about pretrained behavior or a discovered mechanism.
