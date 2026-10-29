# Local broker transport, protocol version 1

This is an authenticated **loopback-only, forward-only** transport for a
trusted, already constructed in-process broker and resident worker. Multiple
local client processes can submit work to one model owner. The broker remains
responsible for fair admission, serial execution, device completion, output
limits, model lifetime and cancellation accounting. The transport does not yet
run traces, interventions, generation, arbitrary Python, model loading or
concurrent GPU kernels. It is not a public or multi-tenant service.

The optional, trusted guardian notification seam is described in
[IPC_CUSTODY_NOTIFICATION.md](IPC_CUSTODY_NOTIFICATION.md). It does not
enable traced IPC jobs or prove result or client-buffer disposal.

## Owner and clients

The owner process creates its trusted model, `ResidentWorker`, and `Broker`,
then starts `IPCServer(broker)`. The server publishes an ephemeral IPv4
`127.0.0.1` address as `server.address`. For identity-bound admission, pass
the fixed, trusted team IDs as `IPCServer(broker, team_ids=["team-a", "team-b"])`.
The server mints a distinct 256-bit bearer secret for each team, exposed to
the owner in the immutable `server.team_tokens` mapping; `server.token` is
`None`. Privately deliver each team only its own token, together with the
address. The team list and tokens last for this server lifetime and cannot be
changed remotely. Restarting requires distributing new tokens.

The legacy `IPCServer(broker)` form instead creates a single shared secret as
`server.token`; it is for trusted cooperative clients. The transport never
writes a discovery file or persists secrets. Convey credentials through an
appropriately private, out-of-band channel. Avoid putting tokens in logs or
command-line arguments. Do not expose the endpoint through a proxy, tunnel or
port forward.

The client does not authenticate the server. If an owner exits and its port is
reused by another local process, a stale client can send that process its
bearer token. Create clients with freshly distributed addresses and tokens for
each server lifetime; discard them when the owner stops or a connection fails,
and do not retry them after an unexpected disconnect. Each completed job closes
its own connection normally. This lifecycle discipline does not eliminate the
port-reuse race. Use the transport only among trusted local processes. A broker
ID sent by an unauthenticated peer would not by itself prove the server's
identity.

```python
# In the model-owning process, after constructing a trusted model:
from looking_glass.broker import Broker
from looking_glass.ipc import IPCServer
from looking_glass.resident_worker import ResidentWorker

model.eval()
worker = ResidentWorker("checkpoint-and-config-id", model, device="cpu")
broker = Broker(worker, max_pending=16, max_pending_per_team=4)
server = IPCServer(broker, team_ids=["team-a", "team-b"]).start()
# Privately share server.address and server.team_tokens[team] with each team.
# Keep this process alive while clients are using the server.
# On shutdown: check server.close(timeout=5); then close/join the broker.
```

Each client uses a fresh connection for one job:

```python
from looking_glass.ipc import IPCClient

client = IPCClient(address, token)  # delivered privately by the owner
handle = client.submit(team_id="team-a", request_id="case-17",
                       inputs={"input_ids": [[10, 20, 30]]}, seed=17)
outcome = handle.wait(timeout=30)
```

Admission and subsequent responses include a broker lifetime `broker_id`,
`job_id`, `team_id`, and `request_id`; terminal results also contain broker
monotonic timestamps. Successful default-encoder results contain a flattened
`logits` list, `shape` (batch, sequence, vocabulary), and `dtype`; they are
not trace events. Record `(broker_id, job_id)` together: `job_id` uniquely
identifies a job only within one broker lifetime, and caller-provided request
IDs may repeat across teams or submissions.
There is no durable registry or result replay after a connection closes.

For finite CPU `torch.float32` results, each JSON number represents the exact
binary32 value as a binary64 number, including subnormals and the sign of zero,
independently of the serializer thread's denormal-flush setting. This is a
numeric wire format, not a raw tensor-byte format: consumers must use a
binary64-capable parser that preserves signed zero. A later conversion back to
float32 may itself flush subnormals, depending on the consumer's environment.
The default encoder also accepts normal values and signed zeros for float16,
bfloat16, and float64; it rejects their subnormals as `invalid_result` until
their transport is independently verified. Other floating dtypes, non-strided
layouts, and lazy negative or conjugate views are rejected. Nonfinite results
are rejected for every supported dtype.
When available, NumPy converts float32 integer bit views in bulk; otherwise a
standard-library decoder provides the same wire values. Conversion can hold
temporary integer copies, NumPy arrays, Python numbers, and JSON buffers at
once. The wire-size preflight is not a reservation or bound on that host memory.

In per-team mode, the server rejects a token used with a different configured
team before broker admission. Token possession is the identity; tokens do not
identify a human or protect against deliberate sharing, compromised accounts,
privileged local processes, or code running in the owner process. In shared
token mode, clients can choose any team label, so fair admission assumes
cooperation. Loopback and bearer secrets are not sufficient isolation for
adversarial users. Traffic has no TLS. Use stronger operating-system isolation
and authenticated encryption before considering such users or network access.

## Framing and validation

Every frame consists of an unsigned four-byte big-endian byte length followed
by exactly that many bytes of UTF-8 JSON. Messages are objects; duplicate keys,
invalid UTF-8, nonfinite values, empty and oversized frames are rejected.
There is no pickle, dynamic import, executable payload, or client-selected
factory. The server sends `ready` only after obtaining a connection slot;
otherwise it sends a `busy` rejection. After `ready`, the client sends exactly
one `submit` object with these exact fields:

```json
{"version":1,"op":"submit","token":"<secret>","team_id":"team-a",
 "request_id":"case-17","job_type":"forward",
 "inputs":{"input_ids":[[10,20,30]]},"seed":17}
```

Identifiers use one to 64 ASCII letters, numbers, underscores, dots or
hyphens. The seed is an integer in `[0, 2**63)`. Inputs are nonempty,
rectangular, equally shaped arrays of nonnegative 32-bit-range integers.
`input_ids` is required; only `attention_mask`, `token_type_ids`, and
`position_ids` may also be supplied. Masks contain only zero or one. Each
dimension is additionally capped by server configuration, and the resident
worker applies its own independent request and model limits. No external
client can supply the trusted request factory or result encoder.

These syntactic checks do not establish that an ID is in a particular
checkpoint's vocabulary, position range, or token-type range. The owner must
validate such bounds against trusted model configuration before broker
admission; the default factory does not do so. An invalid ID passed to a model
can fail a forward and poison the shared worker. Never accept those limits
from a request.

On successful admission the server responds `accepted`; otherwise it returns
`rejected` with a short error code such as `busy`, `queue_full`, `unavailable`,
`unauthorized`, or `invalid_request`. After admission, only a matching
`{"version":1,"op":"cancel","request_id":"case-17"}` message is allowed.
The server confirms cancellation internally and keeps the connection while
waiting for the actual terminal outcome. It sends `succeeded`, `failed`, or
`cancelled`, or sends `timeout` when its job deadline expires. Errors do not
include exception messages, which may contain private input data. The server
checks broker lifetime and outcome identifiers and the default worker result's
request ID before sending values. The client checks the same broker lifetime
on every response after acceptance.

## Limits and cancellation

The default caps are eight accepted connections, 1 MiB per request frame,
4 MiB per response frame, eight batch rows, 256 tokens per row, a three-second
I/O deadline, and a 60-second job deadline. Caps can be reduced or raised
within hard bounds at owner configuration time. Frame bodies are read into a
single bounded buffer; the number of handlers and held handles is bounded by
the connection cap. Slow readers have a bounded write deadline. The worker's
own retained-logit bound also applies, so aggregate completed output depends
on the worker's configured per-result bound and the number of accepted
connections; network bytes alone are not peak host or GPU memory.

Full logits for a large vocabulary or sequence may exceed the response cap.
The default encoder checks an element bound before building Python lists and
checks encoded bytes before sending. Such a job fails with `result_too_large`;
it is never silently truncated. If host allocation fails while serializing,
the transport attempts to report `serialization_out_of_memory`; when even
that small failure response cannot be constructed or sent, the connection can
close without an outcome. Request-local output selection is not yet
supported. Default serialization is intended for small reference workloads,
not high-throughput tensor transfer.

`handle.cancel()` requests cooperative cancellation. `handle.wait()` consumes
the interim acknowledgement and waits for the real terminal outcome, allowing
a later execution or device failure to reach the client. An in-flight kernel
is not preempted. On client disconnect, server deadline, transport failure,
or server shutdown, the server requests cancellation and releases its handle
when the handler exits; the broker retains responsibility for an already
running operation. A `timeout` result means the deadline elapsed and
cancellation was requested: it is **not** proof that work, device memory, or a
lease has stopped. The server job deadline starts after broker submission and
the acceptance response; it does not bound a blocked trusted factory or
broker submission.

The client-side `wait(timeout)` timeout raises `TimeoutError`, closes the
connection and therefore requests cancellation. This differs from
`Broker.RequestHandle.wait(timeout)`, which only times out observation. Once
the transport connection is closed, that client cannot retrieve a later
terminal outcome. Use a sufficiently generous client deadline when the final
worker status matters.

The owner or supervising service must monitor the worker's process-wide
`quarantined` state independently of client connections. If a device drain
fails, retain the device lease and promptly terminate the dedicated execution
process, as required by the resident worker contract. A disconnected or
timed-out client cannot receive that failure, and client-visible errors alone
are not a process-level quarantine signal. This transport does not implement
that supervisor or release a device lease.

`server.close(timeout=...)` stops admission, closes client sockets and
requests cancellation. It returns `False` if startup or any handler has not
exited within the shared deadline; retry to wait again. It does not close the
caller-owned broker or certify that a running worker has finished. If a
context manager cannot finish server shutdown, it raises rather than silently
claiming completion. The broker must be closed and joined separately after
transport shutdown.

To preserve terminal failure delivery during an ordinary worker failure, the
supervisor can call `server.stop_accepting(timeout=...)`, then
`server.drain(timeout=...)` for a bounded grace period, then `server.close()`.
Stopping admission leaves already accepted handlers active and prevents new
submissions; it returns `False` while a broker submission is still in flight.
`drain` requires admission to have been stopped and returns `False` while
handlers remain. A successful drain says only that IPC handlers ended; it does
not certify that a disconnected client's worker or lease has stopped. On a
failed device drain, follow the quarantine/teardown requirement above without
waiting for the grace period.
