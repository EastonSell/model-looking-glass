# GPT-2 trace adapter

This adapter observes eager Hugging Face `GPT2Model` and `GPT2LMHeadModel` executions. It does not load a model, tokenizer, checkpoint, or remote code. Loading and pinning a checkpoint are separate responsibilities. The automated contract tests use a tiny randomly initialized configuration; they are not pretrained-model evidence.

The optional Hugging Face dependency is tested here with `transformers==4.57.3` and PyTorch 2.8.0. Install it in an isolated environment with `python -m pip install transformers==4.57.3` after installing PyTorch. Other Transformers versions have not been validated for this adapter; a dependency import failure is not treated as a passing adapter test.

## What is observed

| Target | Module path for `GPT2LMHeadModel` | Recorded tensor |
| --- | --- | --- |
| `attn` | `transformer.h.{layer}.attn` | Projected attention module output before the block adds its attention residual. This is **not** attention weights. |
| `mlp` | `transformer.h.{layer}.mlp` | MLP module output before the block adds its MLP residual. |
| `block` | `transformer.h.{layer}` | Hidden state after the block's attention and MLP residual additions. |

For `GPT2Model`, the corresponding paths begin with `h.{layer}`. A module may return a tensor or a tuple; the adapter selects the tensor itself (`output_path=()`) or tuple element zero (`output_path=(0,)`), respectively. Each supported output has shape `[batch, local_token, hidden]`. It is a measurement at a named boundary, not evidence of reasoning or a causal explanation. The meaning of these boundaries does not extend automatically to other model families, fused/compiled execution, or altered GPT-2 implementations.

An event records the original output's shape, dtype and device; the exact module path and kind; zero-based layer; forward index within this session; zero-based module-entry invocation count within that forward; and execution-order sequence number. A selected target that fails before its normal capture hook completes invalidates the session, including when its caller catches the exception. A `run_id` is required: use a distinct ID for each session, because event sequence numbers start at zero and duplicate IDs cannot be distinguished. Saved values have shape `[selected_batch, selected_token, hidden]`, with axis order corresponding exactly to the event's ordered `batch_positions` and `token_positions`. Positions are indices in *that module output for that forward*, not global input, cache or generation positions. Batch positions are not stable sample identities across different calls, beam expansion, or reordering. Generation context defaults to unknown and is not inferred from a one-token decode. Prompt token IDs are not saved. `event.to_schema()` returns the shared immutable metadata schema, while the detached tensor remains in `event.value`.

For one explicitly uncached model call, a caller may opt in to `forward_context=GenerationContext(stage="forward", use_cache=False)` using `GenerationContext` from `looking_glass.trace_schema`. This opt-in requires no pre-existing local or global forward/pre-forward hooks, and checks their registration IDs while it runs; it does not attest callback identity. The session requires `use_cache=False` in the actual root call, rejects non-None `past_key_values` and any second forward, and discards the entire trace if these checks fail. It only accepts `input_ids` positionally, so a positional cache cannot bypass this check. This contract assumes exclusive access and unmodified supported GPT-2 execution; it cannot certify transient concurrent hook mutation or monkeypatched implementations. The `forward` label describes that model call, not whether it occurs inside a generation loop. Generation step, absolute cache position, and cache lengths remain unknown, including when `position_ids` are supplied; none are inferred. Other supplied generation contexts are rejected. Default sessions retain their existing multi-forward behavior and unknown context.

## Example

```python
from looking_glass.transformer_trace import GPT2TraceSession

# model is a vanilla, already-loaded GPT2LMHeadModel; input_ids is [batch, token].
model.eval()
trace = GPT2TraceSession(
    model, run_id="short-example", layers=(0,), targets=("attn", "mlp", "block"),
    samples=(0,), tokens=(0, 1), max_events=8, max_trace_bytes=1_000_000,
    capture_device="cpu",
)
with trace:
    output = model(input_ids=input_ids, use_cache=False)

for event in trace.events:
    print(event.metadata())
    selected_values = event.value  # detached snapshot; None in ID-only mode
```

The `direct` mode copies selected values. The `gradient` mode uses the isolated gradient identity in `looking_glass.probes`; it is a comparison instrument, not a gradient of the task loss. Its mathematical identity does not guarantee bitwise equality under every floating-point mode: subnormal values can be flushed during the gradient computation. `id` records metadata only. Value modes reject nonfinite selected values. Hooks do not replace module outputs. Compare traced and uninstrumented behavior under controlled RNG, model mode, dtype and backend; stochastic or nondeterministic execution requires an appropriate prespecified comparison. In particular, a trace cannot demonstrate that every model or training procedure is transparent.

## Limits and execution contract

Sessions are single-use and require exclusive access to the model and its modules. A session is bound to its entering thread, rejects another trace session on overlapping modules, and rejects concurrent/reentrant calls; unrelated callers must not use the same model until the context exits. Hooks are global to the underlying PyTorch modules, not isolated per request. The shared-model dispatcher must own and serialize or safely batch its own hooks; it must not nest or overlap these sessions. Custom hooks that call child modules, external checkpointing/recomputation, compiled models, distributed/sharded models, cross-attention variants, model structure mutation and modified GPT-2 implementations are outside this adapter's contract. Native gradient-checkpointed models are rejected.

The adapter checks event and cumulative retained tensor-byte limits before each capture. Both limits are finite; overflow fails rather than silently truncating. The cumulative byte allowance is not restored if a caller deletes an event. The counter applies only to captured tensor values; it does **not** bound Python metadata, model state, allocator reservation, intermediate tensors, temporary gradient-readout buffers, GPU-to-CPU transfer buffers, or device-wide peak memory. A CPU capture can still allocate temporary GPU tensors and synchronize. This cap alone is not a GPU admission policy.

On an ordinary forward exception, invalid activation, or capture overflow, the session is marked failed and its saved events are discarded immediately. If the exception is caught inside the context, subsequent forwards still fail; registered hooks remain until context exit. An exception leaving the context also discards captured events, and context exit attempts to remove all registered hooks. Cancellation and other `BaseException` cases follow the separate contract below. A failed session must not be interpreted as a complete trace. The exception is propagated; no precision, token selection, or model configuration is silently changed. Capture can expose sensitive data, so snapshots remain in process unless a caller explicitly persists them.

If a hook cannot be removed, the session retains its model lease and `close()` can be retried. If PyTorch registration itself raises before returning a removable handle, or returns an invalid handle, its internal hook state cannot be certified; the model remains quarantined under its lease. Use a newly constructed model or fresh worker process rather than reusing that instance. A cleanup failure is reported without replacing an exception already in flight.

Hook cleanup does not roll back model-side mutations. In particular, a failed or cancelled cached decode can leave a supplied mutable generation cache partially updated across layers. Discard that cache and rebuild it from the original prefix before retrying; do not reuse it as if the failed forward never happened.

Events remain tentative until the context exits and `trace.successful` is true; success confirms the forwards observed by the session, not that every intended call was attempted. Let `KeyboardInterrupt` or `SystemExit` propagate out of the context. PyTorch does not invoke its `always_call` hooks for every `BaseException`; tested unwound-interruption paths invalidate partial traces and clean up, but no Python hook can certify recovery at every asynchronous interruption point. Determining whether an interrupted call has unwound depends on Python stack frames and the stock PyTorch `Module._call_impl`, validated with PyTorch 2.8.0; it does not certify recovery with unavailable frames, monkeypatched or overridden `_call_impl`, compiled execution, or other versions. If such an exception is suppressed inside the context, discard the session regardless of `successful`. An uncertain hook-registration failure quarantines the model as described above; device-level cancellation and recovery belong to the worker/resource manager.

Keep the session handle before entering the context, as in the example. An asynchronous interruption at the return from `__enter__` can leave hooks installed before Python establishes the `with` cleanup. Once the model is quiescent, the entering thread can attempt `trace.close()`; if it fails or the handle was lost, discard the model or worker rather than reuse it.
