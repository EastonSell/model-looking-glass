# Bounded host trace store

Status: a standalone, process-local ownership primitive. It is not yet wired to
the dispatcher, resident worker, IPC, or `MemoryBudget`. Its CPU tests do not
establish GPU memory safety or end-to-end trace admission.

`HostTraceStore` enforces a logical tensor-payload byte cap and an item-count
cap shared across its request handles. Both caps include pending reservations
and copies in progress. A reservation is made before a store-owned copy; if
either cap would be exceeded, `TraceStoreFull` is raised and nothing is
silently dropped. A zero-byte or metadata-only item still occupies a slot.
`usage()` distinguishes claimed totals from ready totals. These totals are
process-local bookkeeping, not a measurement of physical memory. Clients are
trusted in-process code; the private fields are not a security boundary.

Each `store.request(request_id)` returns a distinct lifecycle handle, even if
the text identifier is reused. A request handle is permanently closed by
`cancel()`. Strings used as request or item identifiers must be nonempty and
at most 256 characters; item identifiers may instead be nonnegative integers.
An item identifier may have only one active reservation within a request.
After take or drop it may be reused: old handles cannot access the new item.
Callers producing schema metadata must still use unique run identities where
required by that schema.

```python
import torch
from looking_glass.trace_store import HostTraceStore

store = HostTraceStore(max_tensor_bytes=4096, max_items=4)
request = store.request("run-1")
handle = request.put(0, torch.tensor([1.0, 2.0]))
preview = handle.read()  # independent caller-owned copy, store item remains
owned = handle.take()    # transfer store copy to caller and release store claim
request.cancel()         # close request and discard any remaining items
```

`put()` takes only a plain, dense strided, non-quantized CPU tensor or `None`.
It detaches and clones each tensor to independent contiguous storage; two
aliased inputs produce two separately charged snapshots. It never moves a
tensor from a GPU. Copies are ordinary tensors even under inference mode.
Unsupported tensor subclasses and layouts are rejected
before copying. Callers must not mutate or resize an input during `put()` or
`commit()`; a post-copy check rejects an unexpected shape, dtype, storage size
or alias but cannot make concurrent mutation of a source safe or atomic.
Values are not checked for finiteness by the store.

Use `request.reserve(item_id, nbytes)` when a claim must be visible before
copying or cancellation must target an in-progress copy. Its handle's
`commit(value)` requires exactly the claimed payload bytes and can succeed
only once. A reservation context cancels an unused reservation on normal exit
and rolls back an item on exceptional exit, including after commit. A copy
failure releases the reservation, and a rejected second committer cannot
release the first committer's claim. `handle.drop()` discards a ready item or
cancels a pending one; `handle.cancel()` only cancels an uncommitted item.
`handle.take()` transfers a ready payload and releases its store claim.
`handle.read()` returns an independent copy and leaves the store item in place.

`request.cancel()` removes its ready and reserved items and prevents future
reservations on that handle. If a copy is already in progress, cancellation is
cooperative: it marks the copy for rejection, keeps the reservation charged,
and returns without waiting for the copying thread. That thread must finish or
fail before the claim is released. A caller that needs to establish completion
must join or await its copy operation; aggregate `usage()` alone cannot prove a
particular request has quiesced. A concurrent read and removal serialize
through the store lock. A read that wins first returns a caller-owned copy;
a take that wins first transfers its payload to the caller.

## Boundaries and integration

The cap covers only the logical bytes of store-managed snapshots and their
pending claims. It excludes model inputs and activations, dispatcher-owned
source tensors, metadata and Python objects, allocator rounding/reservations,
temporary cloning workspace, and copies or payloads held by consumers after
`read()` or `take()`. Consequently, concurrent reads or caller-retained outputs
can exceed the store cap in total host memory. Exceptions or framework internals
may also retain tensor references through tracebacks. An interruption during or
immediately after a `take()` or `read()` return can leave a buffer in an
exception graph even when the caller did not receive the value. The store's
internal `take()` return avoids retaining its own transfer local at that
boundary, but cannot clear caller or framework frames. Store bookkeeping is not
an OS memory reservation and cannot prevent out-of-memory errors.

The shared broker must separately claim every live host and GPU buffer through
its `MemoryLease` or another authoritative owner before allocation. In
particular, dispatcher source and store copy overlap during `put`; GPU and host
copies may overlap during offload; each `read()` creates another caller-owned
copy; and a `take()` transfers ownership beyond the store's claim. An
appropriate claim must remain active until each consumer acknowledges
disposal, even if the store usage is already zero. An ambiguous interrupted
handoff requires the separate owner to remain charged until all retained
payload and exception references are disposed or quarantined; cancellation or
a timeout does not prove disposal. The current API neither
creates nor transfers such claims automatically. Do not combine independent
caps into a claim that total system memory is bounded.

Metadata is held separately. A worker must publish metadata and corresponding
store handles together only after model, synchronization and cancellation
postchecks succeed; on a partial failure it must cancel the request and wait
for any pending copy to finish. This primitive alone does not implement that
transaction, transport acknowledgement, or consumer-lifetime accounting.
