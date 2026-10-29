# Pretrained CPU service CLI smoke protocol v1

This opt-in correctness check exercises the real CPU service command in its
own process, using two other authenticated client processes. It complements
the [directly composed worker/broker/IPC check](PRETRAINED_SERVICE_SMOKE.md):
it verifies the runnable owner's startup and shutdown as well as output
isolation. It does not measure GPU performance, throughput or reasoning, and
does not claim that the CPU worker executes two forwards simultaneously.

## Prespecified setup and checks

- Model and tokenizer: `distilbert/distilgpt2` at immutable commit
  `2290a62682d06624634c1f46a6ad5be0f47f38aa`; safetensors SHA-256
  `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
  Both reference and owner use the [offline verified loader](PRETRAINED_MODEL.md).
- Two public fixed single-token inputs, batch one, are sent with seed 1729.
  Their SHA-256 values cover UTF-8 JSON with sorted keys and compact separators:

| Team | Input | Input SHA-256 |
| --- | --- | --- |
| `team-a` | `{"input_ids":[[262]]}` | `44c56ff28f8f656b5baac23de670f27223141ab1df6afac93fe197b9cab44445` |
| `team-b` | `{"input_ids":[[257]]}` | `8ca086bc587bcdca8619f3fc634f4c9d33446366115885bb41161a9da4e36f15` |

- The test constructs serial references using a separate float32 CPU model in
  evaluation mode, eager attention, no generation cache and one PyTorch
  intra-op thread. It releases that model before starting the owner process.
  Thus the reference setup and the owner's resident model do not coexist.
- The owner starts offline with `--max-tokens 1` and two distinct teams.
  Its `OMP_NUM_THREADS` and `MKL_NUM_THREADS` environment settings are both 1;
  the test does not observe the owner's actual PyTorch thread count.
  Credentials travel only through private process pipes and in-memory IPC
  messages; they are never included in summary output or evidence. Two
  distinct client processes use the same caller request ID, and each result
  must retain its own team, broker and unique job identity. A team token must
  be rejected when presented for the other team.
- Every one of the 50,257 float32 output logits must match its own serial
  reference within absolute tolerance `1e-6` and zero relative tolerance.
  The two references must differ by more than `1e-3` somewhere so that a swap
  is detectable. Failed, nonfinite, malformed or oversized outputs fail the
  check; none are shortened or silently retried.
- The test sends `shutdown` and requires a zero exit code with no unexpected
  second stdout frame. Inputs and outputs remain in memory. The check counts
  one explicit reference-model setup and one owner process; it does not claim
  to measure the owner's internal model-load count.

## Run

Install the `transformer` extra and supply a writable cache containing all
pinned files. The test skips when the cache variable is absent; after opt-in,
missing dependencies or assets and all mismatches are failures. It makes no
download request. This PowerShell example hides CUDA devices and sends the
owner's private startup frame only through the test's in-memory pipe:

```powershell
$env:HF_HOME = (Resolve-Path 'work/hf-cache').Path
$env:LOOKING_GLASS_TEST_CACHE = (Join-Path $env:HF_HOME 'hub')
$env:HF_HUB_OFFLINE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONHASHSEED = '0'
python -m unittest discover -s tests -p test_pretrained_cli.py -v
```

`LOOKING_GLASS_CLI_SUMMARY=1` emits aggregate errors, output shape, broker/job
IDs and process/setup counts after clean shutdown. It does not emit tokens,
input text, credentials, logits or activations.
