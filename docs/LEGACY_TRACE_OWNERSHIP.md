# Legacy trace ownership

`TraceSession` reserves the model's registered module tree through the shared
process-local `ModelLease` before installing hooks. Other cooperating hook
sessions must acquire the same lease and reject overlapping roots, including
different model roots that share a submodule. Unrelated models can be traced
independently. The session remains single-use, and its existing event and
logical tensor-byte limits continue to apply.

The lease is an in-process coordination mechanism. It cannot stop custom hooks,
arbitrary model execution, parameter mutation, another process, or direct
changes to a module tree. The tree must remain stable during a session;
acquisition-time paths and identities are checked at entry, before and after
root forwards, at tap callbacks, and when a session closes normally. A change
raises an error; do not use events from an unsuccessful session as a complete
trace. Unlike the newer atomic capture APIs, the public `events` list may still
contain partial records after an exception or capacity overflow. The read-only
`failed` flag stays true after a detected capture, forward, validation, or cleanup
failure. If code catches such a detected failure inside the context, further traced
forwards are rejected and normal context exit raises; use a new session for a
retry. This deliberately changes the previous behavior that permitted recovery
after a caught capture error, including `max_events` and `max_tensor_bytes`
overflow. It also applies to sessions with no tensor-byte limit and to other
capture failures. These checks do not make arbitrary concurrent mutation safe
or isolate requests in a shared-model batch.

Trace callbacks must run on the thread that entered the session. A callback that
reaches an active session on a foreign thread is rejected and marks it failed.
Closing an active session from another thread, or from inside an active root
forward, is also rejected. These checks cannot prevent arbitrary foreign model
calls or detect a callback that was dispatched before cleanup but has not yet
started executing. The caller or broker must exclusively control model
execution and establish quiescence before publishing a trace or reusing the
model. After the owner context has exited, a supervisor may retry a known
cleanup failure from another thread only after ensuring the model is quiescent.
The lease does not synchronize CUDA work; a separate device lease must cover
device completion and cleanup.

Keep a reference to the session before entering it. An interruption can occur
after `__enter__` has completed but before its return value reaches the caller;
in that case hooks and the lease can remain active even though the `with` body
never ran. The entering thread can call `close()` on the retained session after
ensuring model quiescence. If that thread is gone while the session still appears
active, discard the model or worker rather than assuming a supervisor can close
it safely. Do not publish a trace from an interrupted entry.

PyTorch may skip an `always_call` hook when a forward raises a `BaseException`
such as `KeyboardInterrupt` or `SystemExit`. The session may then be unable to
distinguish an unfinished forward from one that already unwound. It fails
closed, keeps its hooks and lease, and rejects `close()` while that state is
uncertain. An uncaught original interruption remains the primary exception, with
the cleanup refusal chained. Whether the interruption was caught or propagated,
discard that model or worker; do not clear internal forward state or publish its
partial events as a completed trace. If code catches the interruption inside the
context, PyTorch may have skipped the callback that would set `failed`, so the
flag can remain false and another forward may still run until exit rejects the
session. A pending root invocation at the next pre-hook can also be legitimate
recursion, so that hook cannot identify this stale state on its own. The caller
must stop using the session immediately upon catching such an interruption and
must not treat any of its events as complete.

The lease is retained until all installed hooks have been removed. If a hook
handle's removal raises, cleanup attempts the other removals, retains the failed
handle and lease, and raises the error. Keep the session object and retry
`close()` after correcting the removal failure; until then, other cooperating
sessions are intentionally rejected. If an experiment or registration failure
and a cleanup failure occur together, the original failure remains primary and
the cleanup failure is chained as its cause. A session whose cleanup fails
must not be treated as a completed trace.

If hook registration raises **before returning a handle**, or returns a value
without a callable removal method, PyTorch may already have installed a hook and
its public API cannot certify removal. The session removes any handles it knows
about but permanently retains the model lease;
`close()` reports that the model or worker must be discarded. This conservative
rule also applies when a custom registration fails before installing anything,
because the caller cannot safely distinguish the two cases. By contrast, if
registration returned a handle and recording or removing it fails, retaining
the session permits a cleanup retry.

These ownership and tree-validation checks add work to the session lifecycle
and capture callbacks. The repository's previously reported CPU timings describe
their original implementation and remain historical results. Do not silently
reuse those measurements as performance claims about this version: prescribe a
new versioned benchmark, save the new source/protocol hashes, and measure again.
