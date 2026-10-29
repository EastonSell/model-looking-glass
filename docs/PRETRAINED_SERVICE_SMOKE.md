# Pretrained shared-service CPU smoke protocol v1

This opt-in integration test checks a small, real pretrained model through the
resident worker, broker and authenticated loopback IPC. It is a correctness
smoke test, not a causal study, latency or throughput benchmark, GPU result,
or evidence that model reasoning has been explained. It deliberately gates
the first forward so both clients can submit while one request is active; the
worker still executes model forwards serially.

## Fixed setup

- Model and tokenizer: `distilbert/distilgpt2`, immutable revision
  `2290a62682d06624634c1f46a6ad5be0f47f38aa`. Weights SHA-256:
  `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
  The [loader](PRETRAINED_MODEL.md) verifies all seven input assets and loads
  offline from the explicit local cache.
- One CPU model load, float32, evaluation mode, eager attention, cache off;
  seed 1729, one PyTorch intra-op thread, and inference mode. The original
  intra-op thread count and RNG state are restored after the test.
- Two public fixed token IDs, batch size one, sequence length one. Hashes
  below cover UTF-8 JSON serialized with sorted keys and compact separators:

| Team | Input | Input SHA-256 |
| --- | --- | --- |
| `team-a` | `{"input_ids":[[262]]}` | `44c56ff28f8f656b5baac23de670f27223141ab1df6afac93fe197b9cab44445` |
| `team-b` | `{"input_ids":[[257]]}` | `8ca086bc587bcdca8619f3fc634f4c9d33446366115885bb41161a9da4e36f15` |

- Worker limits: one example, one token, eight input bytes and 201,028 output
  tensor bytes. The IPC response cap is 2,000,000 bytes; outputs are not
  truncated or retried. Full 50,257-element output logits are compared.
- Acceptance: each client's complete float32 logits must match its own serial
  reference within absolute tolerance `1e-6` and zero relative tolerance.
  The references are computed before broker ownership; the two reference
  outputs must differ by more than `1e-3` somewhere, so a swapped result cannot
  pass both comparisons. Nonfinite, failed, or oversized responses fail.

Two independent client processes use different team tokens and the same
caller request ID. The test checks distinct process and broker job IDs,
matching team/request/broker/job identity in each response, and rejection
when one team uses its token to submit as the other. It verifies one loader
invocation and one Transformers model deserialization, and clean server and
broker shutdown. Inputs and output tensors are held in memory; the test does
not write prompts or activation data.

## Run

Install the optional `transformer` dependency extra and place the pinned
model assets in a writable local cache. Set the cache path explicitly. When
the environment variable is absent the test is skipped; after opting in,
missing dependencies, missing/corrupt cache assets, failures and mismatches
fail the test. No download is attempted.

```powershell
$env:HF_HOME = (Resolve-Path 'work/hf-cache').Path
$env:LOOKING_GLASS_TEST_CACHE = (Join-Path $env:HF_HOME 'hub')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONHASHSEED = '0'
python -m unittest discover -s tests -p test_pretrained_service.py -v
```

Set `LOOKING_GLASS_SERVICE_SUMMARY=1` to print only aggregate error, shape,
broker/job identity and model-load count after successful shutdown. The test
prints no input text, logits, model activations or credentials.

The result applies only to this one checkpoint, input pair, software stack,
CPU, and serial execution setup. It does not establish concurrent CUDA
execution, GPU admission/memory safety, general privacy, or fairness.
