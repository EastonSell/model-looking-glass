# Broker-owned CPU trace IPC

`BrokeredCpuTraceService` connects the bounded [trace IPC v2](TRACE_IPC_V2.md)
to the existing resident `Broker` for two distinct client teams. One broker
owner thread schedules each trace against one `TracedGPT2Worker` and its CPU
model. This first composition allows one request per configured team for the
service lifetime. It is serial: it does not batch requests or establish
simultaneous model execution. The existing forward-only broker and IPC v1
contracts remain unchanged; this service does not mix untraced broker requests
with trace requests.

The trusted operator constructs a fresh CPU `TracedGPT2Worker`, a `HostBudget`,
`HostTraceStore`, `ProducerProfile` and `TraceLimits`, then passes them with two
distinct team IDs and fixed event limits to `BrokeredCpuTraceService`. Construct
one `TraceIPCServer` for that service and distribute its two tokens privately.
Each separate `TraceIPCClient` uses the v2 request, event, receipt and release
protocol. The operator must not also hand the worker, model, underlying trace
service, broker, or internal client capabilities to another scheduler. The
composition rejects another live instance of this same service type using its
worker or model, but it cannot prevent unrelated in-process code from accessing
them. Team tokens authenticate possession; the client does not authenticate the
server. This is a cooperative local boundary, not hostile-code isolation.

The generic broker accepts exact forward-only request and result types. The
private dispatch worker maps each internal scheduling ticket to a rooted,
team-bound trace attempt and runs that attempt on the broker owner thread. Its
zero-byte CPU result is an internal completion token: it is never client logits
and is not itself a trace receipt. The separately validated trace receipt binds
the broker lifetime, job, generation, team, request and trace count. Event
frames continue to carry their own identity and explicit receive/release
messages. A foreign team's receipt is rejected.

## Optional same-forward output

Output delivery is disabled by default. To enable it, the operator gives the
server a positive `max_output_bytes` (at most 64 MiB). A client opts in for a
single request by passing `on_output`, `max_output_bytes`, and the expected
`vocab_size` to `trace`. The server rejects a mismatched vocabulary or a full
`[1, sequence_length, vocabulary]` float32 output larger than either limit or
the worker's output limit before starting a forward. The vocabulary must
match the operator's bound model. Existing trace-only requests and their wire
messages and completion type are unchanged.

The output is the copied CPU logits produced by that same traced invocation;
the internal zero-byte broker result remains only a scheduling token. The
output frame precedes the trace events and carries the broker, process,
job, generation, transfer, team, request and run identities, exact shape,
`torch.float32` dtype, native byte order, byte count, and SHA-256. The client
checks these fields and the received bytes before calling `on_output`. After
the output and every trace event have been released, the completion binds the
same output shape, length, byte order and digest. The digest checks transport
consistency; it does not authenticate the server or independently attest model
execution.

`on_output(metadata, payload)` is synchronous and must return `None`. Its
`bytearray` payload is borrowed only for the duration of the callback. The
callback must drop aliases before returning; any retained copy needs a
separately reserved client-owned budget and lifetime. The sender reserves a
separate serialization buffer before the forward, holds the producer's output
claim while filling and sending it, and releases that buffer only after the
receiver's release acknowledgement. The client separately reserves its
receive buffer before accepting bytes and holds it through the callback.
These cooperative claims do not enforce an RSS limit, cover socket buffers or
prove that a callback actually discarded aliases.

An output callback, transfer or acknowledgement failure, interruption, or
cancellation leaves the accepted attempt incomplete and the uncertain claims
rooted for reconciliation. The output may already have been observed even when
no complete trace receipt is produced; it cannot be retracted or safely
retried on that basis. As with trace-only IPC, a local completion is not proof
that the server received its final acknowledgement or that the whole process
tree drained. The operator must check both server and service close results.

Before the broker copies input, the service reserves and roots a separate
`broker_trace_snapshot` host claim. It keeps that claim through the broker
job, receipt handoff and service shutdown. The input provided by the IPC server
has its own claim; the trace producer, store copy, serialization bytes and
callback-exposure bytes retain their existing claims. Receiver budgets are
separate per client process. These cooperative ledgers do not enforce a combined
process-tree limit or RSS cap and do not include Python object overhead, kernel
socket buffers or unclaimed client-made copies. The callback must release its
borrowed bytes synchronously or separately reserve any persistent copy.
An owner-thread callback cannot submit another trace to this broker; the
service rejects that reentry before admission. A remote callback must also not
wait for another trace on this serial owner: it can block progress until a
deadline and leave one or both requests incomplete.

The operator must stop and successfully close the `TraceIPCServer` before
calling `BrokeredCpuTraceService.close(timeout=...)`. The latter waits for the
broker owner, confirms successful receipts were acknowledged and broker-owned
snapshot tensors are dead, releases the held snapshot claims, and closes the
worker and its own model reference. It does not prove disposal of model aliases
held elsewhere. A `False` result requires a later retry or process-level
reconciliation; it is not proof of disposal. Lock checks and the owner-thread
join use the remaining timeout, subject to the existing broker's own lock behavior.

There is no remote cancellation operation in IPC v2. Broker shutdown can
request cancellation of queued or running tickets, but it cannot preempt a
running model call or retract an event already observed by a client. A
cancelled, disconnected, out-of-memory or otherwise uncertain accepted request
is incomplete and may stop the other team; it is not a clean trace receipt or a
signal to retry. In particular, a job accepted while its peer is busy may run
after its client has disconnected if the server has not observed that loss.
The exact owner, failure graph and outstanding claims remain rooted for trusted
reconciliation or process exit. A local client completion does not prove the
server has drained; check the server's own close result and incomplete status.
A direct receipt decision is ordered with this service's failure latch and
close decision. Recovery and acknowledgment reject known quarantine or closing.
Other health flags are point observations and can change immediately after a
check. A later fault cannot retract an already-decided receipt or acknowledgment,
even if the caller receives it afterward.
A failed or interrupted construction can also retain a started owner; it
requires process-level reconciliation rather than binding another scheduler.

This path is CPU-only and uses an exact float32 GPT-2 worker. It supplies no
GPU admission, trace profile, throughput guarantee, or authorization for the
separately profiled zero-trace GPU service. Any future GPU trace composition
requires its own source-, workload- and device-bound measurement and admission.
