# Local CPU model service

`python -m looking_glass.cpu_service` starts one **CPU-only** owner process
for the pinned DistilGPT2 model and accepts forward requests from separate
local client processes. The owner loads the model once and executes requests
serially; concurrent submission is not concurrent model execution. It uses
the bounded broker and authenticated loopback transport described in
[BROKER.md](BROKER.md) and [IPC.md](IPC.md). There is no GPU use, GPU memory
admission, activation tracing, generation, intervention, or batching in
this CPU entry point.

Install the optional transformer dependencies described by the repository and
prepare the verified pinned checkpoint cache separately. Startup uses the
loader's offline policy: no model files are downloaded by the service. The
service checks the fixed checkpoint and tokenizer file hashes before loading.
If the cache is missing or an asset fails verification, startup fails.

Launch with one or more fixed team names:

```text
python -m looking_glass.cpu_service --cache-dir /path/to/cache --team alpha --team beta
```

The default maximum input length is two tokens. For the pinned DistilGPT2
vocabulary, the current full-logit transport cap permits at most two tokens;
larger `--max-tokens` settings fail at startup before inference. Actual encoded
responses are checked again and can still fail if they exceed the wire limit.
Requests contain token IDs rather than raw text. The service
validates vocabulary and position bounds, input shapes, and binary attention
masks before admission; every mask row must include at least one visible token.

The process writes a single JSON readiness message to stdout containing its
loopback address, broker lifetime ID, model identity, and distinct bearer
tokens for the configured teams. **Treat stdout as a private controller
channel:** do not put the message or tokens in logs, shared files, command
lines, or chat. Token possession authorizes the corresponding team label;
tokens are not a substitute for OS isolation against malicious local users.
Distribute each team only its own token through a private channel.

The controller can keep stdin open while clients run. Writing `shutdown` on a
line requests orderly shutdown; closing stdin has the same effect. The owner
stops admission, requests cancellation of active work, waits for broker and
transport shutdown, and exits nonzero if teardown cannot be verified. A
running forward cannot be preempted, so a timeout is not proof of completion.
On ordinary worker failure, the service briefly drains accepted connections
so they can receive terminal failures; the process still exits nonzero.

A controller can start and use the service without displaying credentials:

```python
import json
import subprocess
import sys
from looking_glass.ipc import IPCClient

owner = subprocess.Popen(
    [sys.executable, '-m', 'looking_glass.cpu_service', '--cache-dir', cache_dir,
     '--team', 'alpha', '--team', 'beta'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)
endpoint = json.loads(owner.stdout.readline())  # private; contains bearer tokens
client = IPCClient(tuple(endpoint['address']), endpoint['team_tokens']['alpha'])
handle = client.submit(team_id='alpha', request_id='example',
                       inputs={'input_ids': [[262, 257]]}, seed=0)
outcome = handle.wait(timeout=60)
owner.stdin.write('shutdown\n')
owner.stdin.flush()
exit_code = owner.wait(timeout=30)
```

Successful results contain flattened logits and their shape. The lifetime
`broker_id` plus `job_id` identifies the request, including when caller
request IDs repeat. The service does not persist a run protocol, input,
result, or artifact registry: experiments must record their own versioned
protocols, source/model identifiers, seeds, controls, failures, and results.

The reusable `serve_owner` lifecycle helper in `looking_glass.service`
supports an externally assembled worker, broker, and IPC server. It monitors
the worker's quarantine signal even during blocked startup and shutdown and
reports workload status separately from `teardown_verified`. The helper is
not itself a GPU service or a process supervisor. A future GPU owner must
use a verified device identity, a process-held device lease and durable
supervisor protocol, and trusted measured resource profiles and live admission.
The owner may attest only after verified shutdown and synchronization, and
must retain its lease through exit. The supervisor may clear the unsafe marker
only after reaping that exact owner and reacquiring the lease.
