# Short-input GPU development pilot: reference and routing preparation

This is preparation for a source-specific GPU observation, not an enabled GPU
owner, admission profile, or throughput benchmark. The separate
[CPU service](SHORT_DISTILGPT2_SERVICE_V1.md) remains the supported prototype.
The two synthetic inputs were reserved in v31; the four science active and
padded inputs and masks were reserved in v33. All six are public, and no
forward is asserted by those reservations. The inventory does not prove a
complete exposure history or pretraining independence.

## CPU reference

The fixed [input manifest](../experiments/short_cpu_service_reference_inputs_v1.json)
has SHA-256 `47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c`.
The collector requires the exact v33 canonical inventory SHA-256
`e586bae02145a62a7163eb030485fa85a33f9e94cd0949a9f88f3f74c7de194b`
and checks each planned token/mask record. Before a real forward, compare the
inputs against any later reservations and observations and preserve the
currentness check. This is a public development reference, not held-out data.

On a clean, independently reviewed public checkout, with the pinned assets
already cached, use a fresh empty bytecode prefix and a new output directory:

```sh
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B -X pycache_prefix=/absolute/fresh/empty/prefix \
  -m looking_glass.short_service_cpu_reference \
  --cache-dir /absolute/hf-hub-cache --inventory /absolute/v33-canonical.json \
  --output-dir /absolute/new/output --git /absolute/git --expected-commit COMMIT
```

The collector verifies the listed committed source blobs, the exact input
manifest and inventory, and at least 2 GiB free on both the output and cache
filesystems before loading. The pinned loader verifies the checkpoint and other assets,
loads CPU float32 eager evaluation mode with cache disabled, and does not
download. It retains all float32 logits for six calls: the two synthetic rows
and each science row both unpadded and padded. Each full tensor is at most
6,432,896 bytes. Start records precede calls and output bytes precede the
finite check; any surviving partial bytes must be retained. A failure or
interrupted write can leave no diagnostic, or can leave a seemingly passing
`candidate.json` even though the call raised. Completion requires a separately
retained, reaped exit code 0, no failure record, and successful full-inventory
hash/readback checks; candidate-file presence alone does not establish success.
For science, the recorded comparison checks all active positions between the
CPU padded and unpadded outputs at `1e-6 + 1e-5 * abs(reference)` elementwise.
The input, source and asset hashes are consistency checks under a trusted,
quiescent process; they do not independently prove which instructions ran.
The disk check is a point observation and is not a host-RAM bound.

## Routing core and future GPU owner

`SerialRoutingPilot` exercises one fixed bucket at a time. It creates a serial
`Broker` over an already-owned `LifetimeWorkspaceWorker`: two reference calls
are followed by two concurrently submitting teams. It preserves the complete
CPU-offloaded logits for every call and checks exact team, request and broker
job identities. For each client, the entire output and the last-active row must
match that client's own serial reference at the fixed tolerance; the two
reference last-active rows must also differ at that tolerance, and neither
client may match the other row. The same-path serial references test routing
and repeatability, not padded-versus-unpadded equivalence. A caller must
retain the owner, inputs, outcomes, client threads and claims through any
uncertain failure. CPU fabricated tests do not authorize a GPU launch.

The core uses in-process Broker clients and full outputs. It does not exercise
IPC, the CPU service's compact last-token projector, concurrent model forwards,
or a two-row batch. A future GPU owner must use one separately supervised
attempt per exact 1×16 or 1×32 shape, with one resident model and two clients
in each attempt. It must bind exact rows, masks, seed, source and asset hashes,
runtime precision and physical identity; acquire the canonical physical lease;
make a fresh shape-specific admission with bounded device, host and disk
commitments; synchronize and retain allocated/reserved peaks and device-free
point samples; preserve raw references and routed tensors; and keep ownership
and claims on uncertain cleanup until supervised process exit. A CPU/GPU
comparison must be explicitly labeled and must not assume bitwise equality.
The 1×2 and grouped 1×4 profiles do not authorize or bound either bucket.
