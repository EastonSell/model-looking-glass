# Serial shared CPU trace service

`CpuSharedTraceService` composes one fresh `CpuTraceOwner` with its existing
sealed-batch handoff. Two trusted in-process clients can submit separate
traced jobs to the same resident CPU GPT-2 worker. Execution and delivery are
serial: each complete job is sealed, delivered to its own client, and drained
before another forward starts. This is a **CPU-tested prototype**, not IPC,
cross-process delivery, authentication, a GPU service, a fairness guarantee,
or evidence of concurrent model execution.

The service returns trace events, not the worker's logits. `team_id` is bound to
a client capability by the trusted owner. `service.client(team_id)` is callable
by code in the same process, so these bindings route cooperative clients; they
do not isolate hostile Python code. The owner must exclusively control the
worker, model, owner and handoff lifecycle, and must authorize that client for
every sample in its own job. Do not separately call the owner or batch while
the service is bound. A CPU owner is bound once for its lifetime; construct a
fresh owner for another service.

## Request and delivery

The trusted operator configures a `ProducerProfile` and `TraceLimits` when
constructing the service. The service snapshots these values; the caller must
not mutate them concurrently during construction. `max_trace_bytes` and
`max_events` are per-job limits, not client-supplied estimates. Existing owner
preflight validates each selected sample/site and its expected payload size
before the forward. An independently supplied `HostBudget` caps the sum of
participating logical host claims and items across clients, producers, stores,
handoff metadata and serialization; `HostTraceStore` has its own additional
copy limits. The exact input and trace-plan limits still apply.

Before allocating inputs, the client reserves a separate input claim from the
same host budget. It must hold that claim and all inputs stable through the
whole `submit` call, **including while waiting** for the other client. The
service does not reserve or dispose caller-owned inputs or their aliases. An
ordinary exception traceback, custom exception attributes or caller frames may
retain input references: the caller must dispose or separately charge all of
them before releasing its input claim. The wrapper makes a best-effort bounded
traceback cleanup on failure; that is not proof of disposal.

```python
import torch
from looking_glass.cpu_shared_trace_service import CpuSharedTraceService
from looking_glass.traced_broker_worker import SampleTrace, SiteTrace

# owner, profile, limits and budget are configured by the trusted operator;
# owner uses an exclusive CPU TracedGPT2Worker and the same budget.
service = CpuSharedTraceService(owner=owner, profile=profile, limits=limits,
                                team_ids=("alpha", "beta"))
client = service.client("alpha")
input_owner = budget.open_request("alpha-input-1")
input_claim = input_owner.reserve("alpha", "input", 24, items=1)
inputs = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}

def consume(manifest, payload):
    # Consume synchronously without retaining manifest, payload or aliases.
    assert manifest.identity.team_id == "alpha"
    assert manifest.identity.request_id == "request-1"
    process_trace_bytes(manifest.event, payload)

receipt = client.submit(request_id="request-1", inputs=inputs,
                        input_owner=input_owner, input_claim=input_claim,
                        samples=(SampleTrace("sample-1", (0,)),),
                        sites=(SiteTrace(0, "attn"),), on_event=consume)
client.acknowledge(receipt)
inputs = None  # also dispose any other input aliases before releasing
input_claim.release()
input_owner.close()
```

`consume` must return `None` synchronously. It must not retain the manifest,
payload, callback frames or their aliases on success; a callback that needs a
retained copy must reserve a separate owner/claim **before** making it. The
same rule applies to additional copies and metadata on failure. The service
reserves a separate per-event `callback_exposure` claim before the callback can
see the serialized bytes. It roots that exact byte object and claim if delivery
or the callback fails. It releases this extra claim only after a successful,
contract-compliant return and dropping its own byte reference. The underlying
delivery may independently dispose its own source, destination or serializer
claims after a failed callback; the extra claim does not cover arbitrary
additional copies or prove the client's disposal.

The callback can see an early event before a later event fails. It must stage
such observations and treat them as a complete trace only after receiving the
returned `TraceReceipt`. A receipt binds the broker/process lifetime, job and
generation, team, request, run, event count and total bytes. It is a local
cooperative completion statement, not an authenticated remote acknowledgement
or proof that another process discarded its copies. The service assigns a
fresh run and increasing job generation; clients must not reuse request IDs
within a service lifetime.

Each team may have one submission or unacknowledged receipt at a time; a second
submission from the active team is rejected. The other team may wait, but there
is no fairness or wait-time guarantee, and a blocked callback blocks the next
job. A lost successful return is recoverable with `client.recover(request_id)`;
it returns the same receipt without running another forward. The owner must
call `client.acknowledge(receipt)` before another submission by that client or
service shutdown. Repeating acknowledgement of that exact most-recent receipt
is idempotent. `close()` refuses active, waiting, quarantined or unacknowledged
work; it closes this coordinator, not the separately owned model/worker.

## Failure and bounds

An exact pre-forward plan, input or known admission rejection is drained and
the service can continue. A `PostForwardCapacityError` means the CPU forward
returned but manifest admission was denied before any trace was delivered; the
staged trace was drained. Record the input exposure and do not treat this as an
unrun request. Other failures, including `HostBudgetExceeded`, can occur after
execution or delivery begins. Unless the caller independently establishes
otherwise, treat them as potentially executed and do not blindly retry.

An uncertain forward, handoff, callback, cleanup or interrupted transition
quarantines the service, prevents subsequent submissions and keeps its exact
active attempt, owner, registries and outstanding claims rooted. There is no
automatic recovery in this slice. A process-lifetime supervisor must retain
those objects and establish quiescence/disposal before any reconciliation, or
replace the process after verified termination. Arbitrary asynchronous
interruption can strand Python locks; a timeout, exception, missing reference,
client acknowledgement or garbage collection does not establish disposal.

The bounds are cooperative logical reservations, **not an RSS or machine-wide
memory limit**. Model parameters and buffers, Python objects, allocator and
library overhead, hidden external backing owners, and unclaimed client copies
are outside these claims. The operator must justify a producer workspace
envelope and any resident-model reserve; the service does not measure them.
These CPU tests establish neither GPU memory admission nor concurrency or
throughput on a device.

The existing [IPC v1](IPC.md) remains forward-only. A cross-process follow-on
requires a separately reviewed, versioned trace protocol with authenticated
team/job/generation routing, bounded metadata and payload framing, reservation
before each encoder/copy, complete-batch publication, per-event receipt and
release semantics, and explicit client/socket custody. A disconnect or ACK
cannot prove another process has discarded a copy.
