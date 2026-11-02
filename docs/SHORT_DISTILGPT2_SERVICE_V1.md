# Short-input DistilGPT2 service: CPU prototype

`short_cpu_service` composes the existing serial `ResidentWorker`, `Broker`
and forward-only IPC v1. It gives one or two trusted local teams a shared CPU
model with a fixed **16- or 32-token bucket**. Each request has one token row
and an explicit right-padded attention mask. A successful response contains
the full vocabulary logits for the **last active token only**, shape
`[1, 1, vocabulary]`. It is not a full-sequence result, a generated token,
traced IPC, multi-request batching, or a GPU service.

Clients may put different active lengths in the same bucket. They use the
same physical tensor shape, but the broker still executes them serially; this
does not demonstrate simultaneous model calls or a throughput improvement.
The trusted owner supplies the model and must prevent concurrent model use or
mutation. Bearer tokens and the loopback transport inherit the limitations in
[IPC v1](https://github.com/EastonSell/model-looking-glass/blob/f3d3e28980379640f7548541225ef40b017e5557/docs/IPC.md); in particular, the client does not authenticate the server.

## Run and submit

With the repository's pinned DistilGPT2 assets already cached, launch:

```sh
python -m looking_glass.cpu_short_service --cache-dir /path/to/hub \
  --bucket 32 --team alpha --team beta
```

Loading is offline and verifies the pinned assets. The owner writes one
endpoint object to its **private controller channel** containing the
per-team credentials, bucket, vocabulary and pad token. Do not publish or log
those credentials. Send `shutdown` to the owner's standard input for normal
termination. Each client privately receives the current address and its own
token. For example, after locally tokenizing a short scientific prompt with the
same pinned tokenizer, a client can submit its nonempty list of IDs:

```python
from looking_glass.ipc import IPCClient
from looking_glass.short_cpu_service import make_short_input

# ids must contain no more than the announced bucket size; no truncation occurs.
inputs = make_short_input(ids, bucket_length=32, vocab_size=50257, pad_token_id=50256)
handle = IPCClient(address, team_token).submit(
    team_id="alpha", request_id="case-1", inputs=inputs, seed=17)
outcome = handle.wait(timeout=60)
assert outcome["status"] == "succeeded"
assert outcome["result"]["shape"] == [1, 1, 50257]
next_token_logits = outcome["result"]["logits"]
```

The list and mask must be exact integer values. Valid masks have at least one
`1`, followed only by `0`s; every masked token ID must be the announced pad
ID. The factory makes bounded row copies, then checks the copied bucket shape,
vocabulary bounds, mask and seed before making request tensors. These
separate row copies do not form an atomic snapshot: callers must not mutate a
request during submission. A later tensor-factory failure is an uncertain
failure, not a clean pre-allocation rejection. Inputs longer than the bucket,
left padding, holes in a mask and an all-zero mask are rejected. The output
position is the last `1` in the mask, even if its ID equals the pad token.
Callers should verify the advertised service version and output shape rather
than treating this response as all token positions.

## Capacity and correctness scope

For the pinned 50,257-entry vocabulary, the underlying model **still creates
and the resident worker copies all logits**: 3,216,448 bytes at 16 positions
or 6,432,896 bytes at 32, before the 201,028-byte compact selected copy. The
last-token tensor is detached, contiguous and independently owned; retaining
it does not retain the full sequence tensor. The 4 MiB IPC response cap is
adequate for the single vocabulary row under the encoder's conservative
32-byte-per-value preflight. It is not a host-memory cap: full logits, model
intermediates, input snapshots, temporary arrays, Python floats and JSON
serialization can overlap. The resident output cap is checked after the
model has allocated logits and is not an advance peak-memory bound. Client
copies and retained results are the clients' responsibility.

CPU tests use a tiny untrained GPT-2 and compare each returned row against a
separate unpadded forward at `atol=1e-6, rtol=1e-5`. They exercise distinct
concurrent clients, active lengths 1, bucket−1 and bucket, routing, malformed
masks and compact output ownership. Passing sampled cases does not prove
padding equivalence for every model, runtime or input. A future pinned-model
reference must use already exposed or separately reserved public inputs and
retain failures and source/runtime/asset identity. The two-token GPU service's
approval does not authorize this workload.

The [fixed reference proposal](../experiments/short_cpu_service_reference_inputs_v1.json)
contains two synthetic bucket-filling rows and two short scientific prompts,
with exact IDs and masks for both the unpadded references and padded service
calls. The scientific active rows are intentional full-row prefixes of their
padded versions; the masks distinguish the proposed executions. Preparation
of this manifest did not include a pretrained forward. Before forwarding any
of these rows, reserve each exact execution in the then-current exposure
inventory and repeat the public overlap check against intervening reservations.
The earlier public-only checks for the [synthetic](SHORT_SERVICE_SYNTHETIC_PUBLIC_SUPPORT_V1.json)
and [scientific](SHORT_SERVICE_SCIENCE_PUBLIC_SUPPORT_V1.json) rows are tied to
v29 and have the scope stated in those receipts; they do not establish complete
exposure history or pretraining independence.

To reconstruct those point-in-time checks, use the public v29 inventory at
`bb220b739bcf144f6a6532abc983b699c1d1aba3`: its
`looking_glass.pretrained_exposures_v29.canonical_inventory_bytes()` produces
SHA-256 `f2dc3ff6cc8befc600cc62de68c9c804f8962c7261b87cad3c94ea7744bb7df6`.
Also obtain the 128 rows from
`experiments/natural_agreement_original_public_support_v1.json` at commit
`8533660206f675b437ad4ea4fb4cef2116956ce1` (file SHA-256
`5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6`),
the 256 `cases[].input_ids` rows from
`experiments/natural_agreement_successor_inputs_v1.json` at commit
`afdbc15317983a3948116c28eb857c5ddb869783` (file SHA-256
`6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4`),
and the 32 `cases[].input_ids` patch rows from
`experiments/distilgpt2_two_attractor_patch_cases_v1.json` at commit
`64d72f4ea3818ea2a33edd9db5073e37417cd5c0` (file SHA-256
`6d0cbba21d15f9ab05df106617141c25a460f07be7ee57c53470f969eb79e518`).
The first support is a conservative public superset conditional on its
documented sampler and custody, rather than a disclosure of sealed membership.
For the v29 input/extension set, include each record's `token_ids` and one
copy of that row followed by each integer `target_token_id`, each integer in
`candidate_token_ids` and each integer in every `candidate_token_groups`
value; the cited receipt counted 3,960
sequences. Compare every proposed row `a` with each reference row `b` using
`a[:len(b)] == b or b[:len(a)] == a`, counting relations in both directions.
The synthetic receipt uses the two unique bucket-filling rows; the science
receipt uses the four active and padded rows. This recipe reconstructs the
earlier comparison, not the mandatory check against newer inventory updates.

## Required GPU measurement and admission work

There is **no GPU capacity claim or launch authorization** in this prototype.
The current fixed owner admits only its exact one-row/two-token, no-mask,
full-output signature. Do not widen its shape or reuse its load/cold/warm
profile to admit this service. A separate reviewed, source-bound owner and
coordinated device slot are required before any GPU work.

1. Freeze the exact owner, worker, short-request and projection code, input
   and output schema, checkpoint/config/tokenizer asset hashes, code and
   dependency versions, runtime/numerical settings, device UUID and physical
   lease identity. Bind `input_ids` **and** `attention_mask`, one-row shape,
   float32 eager evaluation, cache disabled, no tracing, output selection and
   serialization to the new workload identity. Source changes invalidate a
   source-specific profile. At minimum, inventory this prototype's
   `short_cpu_service.py`, `cpu_short_service.py`, `resident_worker.py`,
   `broker.py`, `ipc.py`, `service.py`, `pretrained.py` and model-ownership
   dependencies, plus every new GPU owner, supervisor, lease, admission and
   host-projection file actually used. The eventual measured source may
   differ from this CPU-only prototype and must be reviewed as its own tree.
2. Collect independently reviewed load, cold and warm observations for each
   actual 16- and 32-position shape, including representative short and full
   masks. A two-row grouped call would need separate evidence and admission;
   this prototype makes no grouped call. Synchronize around observations and
   record actual shape/forward identity, PyTorch allocated/reserved peaks,
   device-wide free/total point samples, and the model/worker exit and cleanup
   evidence. Point samples are not a device-wide peak or a proof that other
   processes were idle.
3. Account for the full GPU logits and intermediates, full CPU offload,
   compact copy, input snapshots, at most two pending clients, serializer
   buffers, and client/result retention. Measure and justify an independent
   host envelope; neither logical tensor bytes nor a wire limit establishes
   an RSS bound. The post-worker CPU projection and its failure/cancellation
   custody must be integrated with the GPU owner's completion authority before
   any result or cleanup attestation.
4. Under the physical lease, perform fresh runtime, identity, allocator and
   device/host headroom checks against approved **matching** profiles and
   separate resident and workspace commitments. Retain partial evidence and
   quarantine uncertain completion; require the supervising process to observe
   child exit and reconcile markers before releasing the slot. A sampled peak
   is evidence for its observed workload, not a universal future memory bound.
