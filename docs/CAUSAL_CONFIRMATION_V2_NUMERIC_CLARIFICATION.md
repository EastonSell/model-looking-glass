# Synthetic causal v2: pre-seal CPU numeric clarification

This clarification supplements the proposed, pre-seal
[v2 protocol](CAUSAL_CONFIRMATION_V2.md); it does not alter the recorded v1
protocol or results. It must be committed, independently reviewed and bound to
the public pre-manifest before any v2 held-out set or site schedule is sealed.
The pre-manifest records this file's SHA-256 in a separate
`numeric_clarification_sha256` field. The existing v2 `context` object keeps
its prescribed exact keys; its `source_revision` must include this committed
clarification. A missing or mismatched clarification invalidates preparation.

In addition to v2's CPU float32, one-intra-op-thread and `torch.no_grad()`
requirements, each fresh execution process must request denormal preservation
with `torch.set_flush_denormal(False)` and require a successful return before
constructing the model or performing any model forward. Immediately verify
both of these probes on the execution thread, and repeat them after setting
the intra-op thread count and before the first model forward. Set the count
to one before constructing or reinitializing any study model:

1. Construct an int32 tensor containing `1`, reinterpret its bits as float32,
   and verify that its value does **not** compare equal to float32 zero. This
   checks the minimum positive float32 subnormal as an input to a comparison.
2. Multiply the smallest positive normal float32 by `0.5`, reinterpret the
   result as int32, and require exactly `0x00400000`. This separately checks
   production of a subnormal result.

Record the PyTorch version/build, CPU architecture, policy request result,
probe results, process identity and thread policy with the run. A failed or
unsupported policy/probe is a failed preparation or invalid execution, not a
reason to continue with silently changed arithmetic. If failure occurs after a
run record exists, retain its failure and partial evidence. The policy applies
to the actual execution thread; if a future implementation uses additional
threads, each thread that performs a model forward or relevant tensor
comparison must establish and verify the same policy before work. The current
single-thread protocol does not claim a process-wide setting for arbitrary
other threads or devices.

PyTorch numeric equality is not automatically bitwise equality under every
floating-point environment: flushing can make a nonzero subnormal compare
equal to zero. All exact comparisons required by v2—including sparse trace
values, full control logits, and private injection-integrity tensors—must reject
nonfinite values, inspect float32 bit patterns, and explicitly treat positive
and negative zero as equivalent. The execution-policy checks remain necessary
because a comparator cannot retroactively standardize model forward arithmetic.
These measures support the stated CPU execution contract and do not imply
cross-platform bitwise reproducibility.
